"""API 端到端测试（FastAPI TestClient + 临时 SQLite）。"""

import os
import tempfile

import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["SVGAPI_DB_PATH"] = _tmp.name

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_lifecycle(client):
    body = {
        "d": "M0,0h40v40h-40Z M10,10h20v20h-20Z",
        "fill_rule": "nonzero",
        "metadata": {"part": "法兰", "material": "Q235"},
        "contour_metadata": [{"name": "外法兰"}, {"name": "螺栓孔"}],
    }
    r = client.post("/paths", json=body)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    assert r.json()["metadata"]["part"] == "法兰"

    r = client.post(f"/paths/{pid}/normalize")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["normalized"]["fill_rule"] == "nonzero"
    contours = data["normalized"]["contours"]
    assert contours[0]["role"] == "outer"
    assert contours[1]["role"] == "hole"
    assert contours[0]["metadata"] == {"name": "外法兰"}
    assert contours[1]["bounds"]["minx"] == 10

    r = client.get(f"/paths/{pid}")
    assert r.status_code == 200
    got = r.json()
    assert got["d"] == body["d"]
    assert got["normalized"]["d"] == data["normalized"]["d"]
    assert isinstance(got["normalized"]["warnings"], list)
    assert got["metadata"]["material"] == "Q235"


def test_get_unormalized_has_null(client):
    r = client.post("/paths", json={"d": "M0,0h10v10h-10Z"})
    pid = r.json()["id"]
    r = client.get(f"/paths/{pid}")
    assert r.status_code == 200
    assert r.json()["normalized"] is None


def test_404(client):
    assert client.get("/paths/999999").status_code == 404
    assert client.post("/paths/999999/normalize").status_code == 404


def test_invalid_d_rejected(client):
    r = client.post("/paths", json={"d": "M0,0 L", "fill_rule": "nonzero"})
    assert r.status_code == 422


def test_invalid_fill_rule_rejected(client):
    r = client.post("/paths",
                    json={"d": "M0,0h10Z", "fill_rule": "winding"})
    assert r.status_code == 422


def test_open_contour_warning_surfaced(client):
    r = client.post("/paths", json={"d": "M0,0 L40,0 L40,40"})
    pid = r.json()["id"]
    r = client.post(f"/paths/{pid}/normalize")
    warnings = r.json()["normalized"]["warnings"]
    assert any(w["code"] == "OPEN_CONTOUR" for w in warnings)


def test_repeated_normalize_idempotent(client):
    r = client.post("/paths",
                    json={"d": "M0,0h40v40h-40Z M10,10h20v20h-20Z"})
    pid = r.json()["id"]
    d1 = client.post(f"/paths/{pid}/normalize").json()["normalized"]["d"]
    d2 = client.post(f"/paths/{pid}/normalize").json()["normalized"]["d"]
    assert d1 == d2


# ---------------------------------------------------------------------------
# POST /paths/compare 版本差异审计
# ---------------------------------------------------------------------------

def _mk(client, d, **kw):
    return client.post("/paths", json={"d": d, **kw}).json()["id"]


def test_compare_identical_is_empty(client):
    pid = _mk(client, "M0,0h40v40h-40Z M10,10h20v20h-20Z")
    r = client.post("/paths/compare", json={"base_id": pid, "candidate_id": pid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == [] and body["removed"] == [] and body["modified"] == []
    assert body["summary"] == {
        "base_contour_count": 2, "candidate_contour_count": 2,
        "added": 0, "removed": 0, "modified": 0, "unchanged": 2}


def test_compare_orientation_only_change(client):
    base = _mk(client, "M0,0h40v40h-40Z M10,10h20v20h-20Z")
    # 内孔同一几何、逆时针书写
    cand = _mk(client, "M0,0h40v40h-40Z M10,30h20v-20h-20Z")
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": cand}).json()
    assert len(body["modified"]) == 1
    m = body["modified"][0]
    assert m["change_type"] == "orientation"
    assert m["matched_by"] == "fingerprint"
    assert m["fingerprint"]["base"] == m["fingerprint"]["candidate"]
    assert m["base"]["original_index"] == 1
    assert m["candidate"]["original_index"] == 1
    assert m["base"]["role"] == m["candidate"]["role"] == "hole"
    assert m["base"]["depth"] == m["candidate"]["depth"] == 1
    assert m["base"]["metadata"] == {}


def test_compare_hierarchy_change_takes_priority_over_orientation(client):
    # 基线：大外环 + 内孔（孔顺时针，面积为正）
    base = _mk(client, "M0,0h100v100h-100Z M10,10h20v20h-20Z")
    # 候选：大外环消失，同一小环成为顶层外环（仍顺时针）-> 层级变化
    cand = _mk(client, "M10,10h20v20h-20Z")
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": cand}).json()
    assert len(body["removed"]) == 1
    assert body["removed"][0]["base_original_index"] == 0
    assert body["removed"][0]["base_role"] == "outer"
    assert len(body["modified"]) == 1
    m = body["modified"][0]
    assert m["change_type"] == "hierarchy"
    assert m["matched_by"] == "fingerprint"
    assert m["base"]["role"] == "hole" and m["base"]["depth"] == 1
    assert m["candidate"]["role"] == "outer" and m["candidate"]["depth"] == 0


def test_compare_geometry_change_matched_by_original_index(client):
    base = _mk(client, "M0,0h40v40h-40Z M10,10h20v20h-20Z")
    cand = _mk(client, "M0,0h40v40h-40Z M12,12h16v16h-16Z")
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": cand}).json()
    assert len(body["modified"]) == 1
    m = body["modified"][0]
    assert m["change_type"] == "geometry"
    assert m["matched_by"] == "index"
    assert m["fingerprint"]["base"] != m["fingerprint"]["candidate"]
    assert m["base"]["original_index"] == m["candidate"]["original_index"] == 1
    assert m["base"]["role"] == m["candidate"]["role"] == "hole"


def test_compare_same_fingerprint_but_index_changed_is_unchanged(client):
    # 子路径顺序重排：相同指纹、原始索引互换，仍视为同一轮廓
    base = _mk(client, "M0,0h40v40h-40Z M10,10h20v20h-20Z")
    cand = _mk(client, "M10,10h20v20h-20Z M0,0h40v40h-40Z")
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": cand}).json()
    assert body["modified"] == []
    assert body["added"] == [] and body["removed"] == []
    assert body["summary"]["unchanged"] == 2


def test_compare_added_and_removed(client):
    base = _mk(client, "M0,0h40v40h-40Z M10,10h20v20h-20Z")
    cand = _mk(client,
               "M0,0h40v40h-40Z M30,30h5v5h-5Z M10,10h20v20h-20Z")
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": cand}).json()
    # 原有两轮廓按指纹匹配为 unchanged，新增一个孔；新增项按候选索引升序
    assert body["summary"] == {
        "base_contour_count": 2, "candidate_contour_count": 3,
        "added": 1, "removed": 0, "modified": 0, "unchanged": 2}
    added = body["added"][0]
    assert added["candidate_original_index"] == 1
    assert added["candidate_role"] == "hole"
    assert added["candidate_depth"] == 1
    assert len(added["candidate_fingerprint"]) == 64

    # 反向比较：同一条轮廓变为 removed
    rev = client.post("/paths/compare",
                      json={"base_id": cand, "candidate_id": base}).json()
    assert rev["removed"][0]["base_original_index"] == 1
    assert rev["summary"]["removed"] == 1 and rev["summary"]["added"] == 0


def test_compare_identical_duplicate_contours_pair_stably(client):
    # 两条坐标完全相同的孔（同指纹、不同索引）：新增第三条同形孔时只计一个 added
    hole = "M10,10h10v10h-10Z"
    base = _mk(client, f"M0,0h100v100h-100Z {hole} {hole}")
    cand = _mk(client, f"M0,0h100v100h-100Z {hole} {hole} {hole}")
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": cand}).json()
    assert body["summary"]["added"] == 1
    assert body["summary"]["unchanged"] == 3
    assert body["modified"] == [] and body["removed"] == []

    # 重排后（孔与外环交换位置）仍全部识别为同一轮廓
    reordered = _mk(client, f"{hole} M0,0h100v100h-100Z {hole}")
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": reordered}).json()
    assert body["summary"]["unchanged"] == 3
    assert body["added"] == [] and body["removed"] == [] and body["modified"] == []


def test_compare_carries_metadata_both_sides(client):
    base = _mk(client, "M0,0h40v40h-40Z M10,10h20v20h-20Z",
               contour_metadata=[{"name": "壳体-v1"}, {"name": "轴孔-v1"}])
    cand = _mk(client, "M0,0h40v40h-40Z M12,12h16v16h-16Z",
               contour_metadata=[{"name": "壳体-v2"}, {"name": "轴孔-v2"}])
    body = client.post("/paths/compare",
                       json={"base_id": base, "candidate_id": cand}).json()
    m = body["modified"][0]
    assert m["base"]["metadata"] == {"name": "轴孔-v1"}
    assert m["candidate"]["metadata"] == {"name": "轴孔-v2"}
    # 未变化的轮廓不作为 modified 返回
    assert len(body["modified"]) == 1


def test_compare_does_not_require_prior_normalization(client):
    # 写入后不调用 /normalize，compare 仍可即时规范化
    pid = _mk(client, "M0,0h10v10h-10Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    r = client.post("/paths/compare", json={"base_id": pid, "candidate_id": pid})
    assert r.status_code == 200
    # 审计不落库、不产生 normalized 结果
    assert client.get(f"/paths/{pid}").json()["normalized"] is None


def test_compare_404_when_either_path_missing(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    r = client.post("/paths/compare", json={"base_id": 999999, "candidate_id": pid})
    assert r.status_code == 404
    assert "base_id" in r.json()["detail"]
    r = client.post("/paths/compare", json={"base_id": pid, "candidate_id": 999999})
    assert r.status_code == 404
    assert "candidate_id" in r.json()["detail"]


def test_compare_422_on_invalid_params(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    assert client.post("/paths/compare",
                       json={"base_id": 0, "candidate_id": pid}).status_code == 422
    assert client.post("/paths/compare",
                       json={"candidate_id": pid}).status_code == 422
    assert client.post("/paths/compare",
                       json={"base_id": "x", "candidate_id": pid}).status_code == 422


def test_compare_422_when_stored_path_unparseable(client):
    # 直接落库一条无法解析的历史脏数据
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    good = _mk(client, "M0,0h10v10h-10Z")
    r = client.post("/paths/compare", json={"base_id": bad_id, "candidate_id": good})
    assert r.status_code == 422
    assert "base_id" in r.json()["detail"]
    r = client.post("/paths/compare", json={"base_id": good, "candidate_id": bad_id})
    assert r.status_code == 422
    assert "candidate_id" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /paths/{id}/clearance 轮廓间安全间隙校验
# ---------------------------------------------------------------------------

def test_clearance_flags_hole_parent_conflict(client):
    # 外方 0..40，内孔 5..35：最小间隙 5，孔的直接 parent 即外环
    pid = _mk(client, "M0,0h40v40h-40Z M5,5h30v30h-30Z")
    r = client.post(f"/paths/{pid}/clearance", json={"min_distance": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid
    assert body["min_distance"] == 8.0
    assert body["include_open"] is False
    assert body["checked_contour_count"] == 2
    assert body["skipped_contours"] == []
    assert body["conflict_count"] == 1
    cf = body["conflicts"][0]
    assert cf["a"] == {"original_index": 0, "role": "outer"}
    assert cf["b"] == {"original_index": 1, "role": "hole"}
    assert cf["distance"] == 5.0
    assert cf["threshold"] == 8.0
    assert cf["hole_parent"] is True


def test_clearance_equal_threshold_passes(client):
    pid = _mk(client, "M0,0h40v40h-40Z M5,5h30v30h-30Z")
    body = client.post(f"/paths/{pid}/clearance",
                       json={"min_distance": 5}).json()
    assert body["conflicts"] == [] and body["conflict_count"] == 0


def test_clearance_conflicts_sorted_by_original_index(client):
    # 三个轮廓，冲突对应按 (小索引, 大索引) 确定性升序
    d = ("M0,0h100v100h-100Z "
         "M10,10h20v20h-20Z "
         "M31,10h20v20h-20Z")
    pid = _mk(client, d)
    body = client.post(f"/paths/{pid}/clearance",
                       json={"min_distance": 1.5}).json()
    keys = [(c["a"]["original_index"], c["b"]["original_index"])
            for c in body["conflicts"]]
    assert keys == sorted(keys)
    assert keys == [(1, 2)]
    # 同级孔之间不是直接 parent 关系
    assert body["conflicts"][0]["hole_parent"] is False


def test_clearance_skips_open_by_default_and_includes_on_request(client):
    d = "M0,0h20v20h-20Z M21,5 L22,6 L21,7"
    pid = _mk(client, d)

    body = client.post(f"/paths/{pid}/clearance",
                       json={"min_distance": 2}).json()
    assert body["checked_contour_count"] == 1
    assert body["conflict_count"] == 0
    assert body["skipped_contours"] == [
        {"original_index": 1, "role": "open", "reason": "open"}]

    body = client.post(f"/paths/{pid}/clearance",
                       json={"min_distance": 2, "include_open": True}).json()
    assert body["include_open"] is True
    assert body["checked_contour_count"] == 2
    assert body["skipped_contours"] == []
    assert body["conflict_count"] == 1
    cf = body["conflicts"][0]
    assert cf["b"] == {"original_index": 1, "role": "open"}
    assert cf["distance"] == 1.0
    assert cf["hole_parent"] is False


def test_clearance_always_skips_degenerate(client):
    d = "M0,0h40v40h-40Z M5,5 L10,5 L5,5 Z M20,20 L25,20"
    pid = _mk(client, d)
    body = client.post(f"/paths/{pid}/clearance",
                       json={"min_distance": 1, "include_open": True}).json()
    assert body["checked_contour_count"] == 1
    reasons = {(s["original_index"], s["reason"])
               for s in body["skipped_contours"]}
    assert reasons == {(1, "degenerate"), (2, "degenerate")}


def test_clearance_is_readonly_and_needs_no_prior_normalization(client):
    pid = _mk(client, "M0,0h40v40h-40Z M5,5h30v30h-30Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    assert client.post(f"/paths/{pid}/clearance",
                       json={"min_distance": 8}).status_code == 200
    # 即时校验不写库
    assert client.get(f"/paths/{pid}").json()["normalized"] is None


def test_clearance_deterministic_across_calls(client):
    d = ("M0,0h100v100h-100Z M10,10h20v20h-20Z M31,10h20v20h-20Z "
         "M60,60h10v10h-10Z")
    pid = _mk(client, d)
    bodies = [client.post(f"/paths/{pid}/clearance",
                          json={"min_distance": 5}).json() for _ in range(3)]
    assert bodies[0] == bodies[1] == bodies[2]


def test_clearance_404(client):
    assert client.post("/paths/999999/clearance",
                       json={"min_distance": 1}).status_code == 404


def test_clearance_422_on_invalid_params(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    url = f"/paths/{pid}/clearance"
    assert client.post(url, json={"min_distance": 0}).status_code == 422
    assert client.post(url, json={"min_distance": -3}).status_code == 422
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"min_distance": "x"}).status_code == 422
    assert client.post(url, json={"min_distance": "nan"}).status_code == 422
    # 超大数（JSON 解析为 Infinity）也必须是 422 而不是 500
    r = client.post(url,
                    content='{"min_distance": 1e999}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_clearance_422_on_unparseable_stored_path(client):
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    r = client.post(f"/paths/{bad_id}/clearance", json={"min_distance": 1})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /paths/{id}/metrics 填充面积与面积质心（只读）
# ---------------------------------------------------------------------------

def test_metrics_simple_outer(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    r = client.post(f"/paths/{pid}/metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid
    assert body["fill_rule"] == "nonzero"
    assert body["area"] == 100.0
    assert body["centroid"] == {"x": 5.0, "y": 5.0}
    assert body["contributing_contour_count"] == 1
    assert body["contours"][0]["role"] == "outer"
    assert body["skipped_contours"] == []


def test_metrics_outer_hole_island(client):
    # 100 - 64 + 4 = 40；质心偏向 (7,7) 的岛 -> 5.2
    pid = _mk(client,
              "M0,0h10v10h-10Z M1,1h8v8h-8Z M6,6h2v2h-2Z")
    body = client.post(f"/paths/{pid}/metrics").json()
    assert body["area"] == 40.0
    assert abs(body["centroid"]["x"] - 5.2) < 1e-9
    assert abs(body["centroid"]["y"] - 5.2) < 1e-9
    assert body["contributing_contour_count"] == 3
    assert [c["role"] for c in body["contours"]] == ["outer", "hole", "outer"]
    assert [c["depth"] for c in body["contours"]] == [0, 1, 2]


def test_metrics_orientation_independent(client):
    # 外环逆时针书写，面积仍为正
    pid = _mk(client, "M0,0 L0,10 L10,10 L10,0 Z")
    body = client.post(f"/paths/{pid}/metrics").json()
    assert body["area"] == 100.0
    assert body["centroid"] == {"x": 5.0, "y": 5.0}


def test_metrics_skips_open_and_degenerate(client):
    d = "M0,0h10v10h-10Z M5,5 L8,5 L5,5 Z M20,0 L30,0 L30,10"
    pid = _mk(client, d)
    body = client.post(f"/paths/{pid}/metrics").json()
    assert body["area"] == 100.0
    assert body["contributing_contour_count"] == 1
    reasons = {(s["original_index"], s["reason"])
               for s in body["skipped_contours"]}
    assert reasons == {(1, "degenerate"), (2, "open")}


def test_metrics_empty_path_area_zero_centroid_null(client):
    pid = _mk(client, "")
    body = client.post(f"/paths/{pid}/metrics").json()
    assert body["area"] == 0.0
    assert body["centroid"] is None
    assert body["contributing_contour_count"] == 0


def test_metrics_only_open_area_zero_centroid_null(client):
    pid = _mk(client, "M0,0 L10,0 L10,10")
    body = client.post(f"/paths/{pid}/metrics").json()
    assert body["area"] == 0.0
    assert body["centroid"] is None
    assert body["skipped_contours"][0]["reason"] == "open"


def test_metrics_curve_flattened_area(client):
    # 细分多边形近似：单位圆面积在 (3, π) 之间，质心为原点
    import math
    pid = _mk(client, "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z")
    body = client.post(f"/paths/{pid}/metrics").json()
    assert 3.0 < body["area"] < math.pi
    assert abs(body["centroid"]["x"]) < 1e-9
    assert abs(body["centroid"]["y"]) < 1e-9


def test_metrics_is_readonly_and_needs_no_prior_normalization(client):
    pid = _mk(client, "M0,0h10v10h-10Z M2,2h6v6h-6Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    assert client.post(f"/paths/{pid}/metrics").status_code == 200
    # 只读：不产生规范化结果，重复调用一致
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    b1 = client.post(f"/paths/{pid}/metrics").json()
    b2 = client.post(f"/paths/{pid}/metrics").json()
    assert b1 == b2


def test_metrics_404(client):
    r = client.post("/paths/999999/metrics")
    assert r.status_code == 404


def test_metrics_422_on_unparseable_stored_path(client):
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    r = client.post(f"/paths/{bad_id}/metrics")
    assert r.status_code == 422
    assert "无法解析" in r.json()["detail"]


def test_metrics_id_path_validation(client):
    # 非整数 id 走 FastAPI 422；数字 0 是合法 int，查无此路径 -> 404
    assert client.post("/paths/x/metrics").status_code == 422
    assert client.post("/paths/0/metrics").status_code == 404


# ---------------------------------------------------------------------------
# POST /paths/{id}/scanlines 水平扫描线填充区间（只读）
# ---------------------------------------------------------------------------

def test_scanlines_simple_outer(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    r = client.post(f"/paths/{pid}/scanlines", json={"spacing": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid
    assert body["fill_rule"] == "nonzero"
    assert body["spacing"] == 5.0
    assert body["bounds"] == {"miny": 0.0, "maxy": 10.0}
    assert body["scanline_count"] == 2
    assert body["lines"] == [
        {"y": 0.0, "intervals": [[0.0, 10.0]]},
        {"y": 5.0, "intervals": [[0.0, 10.0]]},
    ]
    assert body["skipped_contours"] == []


def test_scanlines_outer_hole_island(client):
    pid = _mk(client,
              "M0,0h10v10h-10Z M2,2h6v6h-6Z M6,6h2v2h-2Z")
    body = client.post(f"/paths/{pid}/scanlines",
                       json={"spacing": 2}).json()
    by_y = {line["y"]: line["intervals"] for line in body["lines"]}
    assert by_y[0.0] == [[0.0, 10.0]]
    assert by_y[2.0] == [[0.0, 2.0], [8.0, 10.0]]
    assert by_y[6.0] == [[0.0, 2.0], [6.0, 10.0]]


def test_scanlines_skips_open_and_degenerate(client):
    d = "M0,0h10v10h-10Z M5,5 L8,5 L5,5 Z M20,0 L30,0 L30,10"
    pid = _mk(client, d)
    body = client.post(f"/paths/{pid}/scanlines",
                       json={"spacing": 5}).json()
    assert body["scanline_count"] == 2
    reasons = {(s["original_index"], s["reason"])
               for s in body["skipped_contours"]}
    assert reasons == {(1, "degenerate"), (2, "open")}


def test_scanlines_no_valid_contours_empty_list(client):
    pid = _mk(client, "M0,0 L10,0 L10,10")
    body = client.post(f"/paths/{pid}/scanlines",
                       json={"spacing": 1}).json()
    assert body["lines"] == []
    assert body["scanline_count"] == 0
    assert body["bounds"] is None
    assert body["skipped_contours"][0]["reason"] == "open"


def test_scanlines_422_when_over_1000(client):
    pid = _mk(client, "M0,0h1v10h-1Z")
    r = client.post(f"/paths/{pid}/scanlines", json={"spacing": 0.005})
    assert r.status_code == 422
    assert "1000" in r.json()["detail"]
    # 恰好 1000 条（4/0.004）允许
    pid2 = _mk(client, "M0,0h4v4h-4Z")
    r = client.post(f"/paths/{pid2}/scanlines", json={"spacing": 0.004})
    assert r.status_code == 200
    assert r.json()["scanline_count"] == 1000


def test_scanlines_422_on_invalid_params(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    url = f"/paths/{pid}/scanlines"
    assert client.post(url, json={"spacing": 0}).status_code == 422
    assert client.post(url, json={"spacing": -3}).status_code == 422
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"spacing": "x"}).status_code == 422
    assert client.post(url, json={"spacing": "nan"}).status_code == 422
    r = client.post(url,
                    content='{"spacing": 1e999}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_scanlines_is_readonly_and_needs_no_prior_normalization(client):
    pid = _mk(client, "M0,0h10v10h-10Z M2,2h6v6h-6Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    assert client.post(f"/paths/{pid}/scanlines",
                       json={"spacing": 2}).status_code == 200
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    b1 = client.post(f"/paths/{pid}/scanlines",
                     json={"spacing": 2}).json()
    b2 = client.post(f"/paths/{pid}/scanlines",
                     json={"spacing": 2}).json()
    assert b1 == b2


def test_scanlines_404(client):
    assert client.post("/paths/999999/scanlines",
                       json={"spacing": 1}).status_code == 404


def test_scanlines_422_on_unparseable_stored_path(client):
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    r = client.post(f"/paths/{bad_id}/scanlines", json={"spacing": 1})
    assert r.status_code == 422
    assert "无法解析" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /paths/{id}/cut-plan 闭合轮廓加工计划（只读）
# ---------------------------------------------------------------------------

def test_cut_plan_hole_before_outer_with_air_distances(client):
    # 外方 0..20 含内孔 5..15；起点 (0,0)
    pid = _mk(client, "M0,0h20v20h-20Z M5,5h10v10h-10Z")
    r = client.post(f"/paths/{pid}/cut-plan", json={"x": 0, "y": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid
    assert body["fill_rule"] == "nonzero"
    assert body["start"] == {"x": 0.0, "y": 0.0}
    assert body["total_contours"] == 2
    assert body["skipped_contours"] == []

    s1, s2 = body["steps"]
    # 后代先加工：孔（索引 1）先于外环（索引 0）
    assert s1["order"] == 1 and s2["order"] == 2
    assert s1["original_index"] == 1 and s1["role"] == "hole"
    assert s1["depth"] == 1
    assert s1["entry_point"] == {"x": 5.0, "y": 5.0}
    assert s1["entry_vertex"] == 0
    assert abs(s1["air_distance"] - (5 * 2 ** 0.5)) < 1e-9
    assert s1["head_after"] == s1["entry_point"]
    # 第二步刀头停在 (5,5)，外环最近顶点 (0,0)
    assert s2["original_index"] == 0 and s2["role"] == "outer"
    assert s2["entry_point"] == {"x": 0.0, "y": 0.0}
    assert abs(s2["air_distance"] - (5 * 2 ** 0.5)) < 1e-9
    assert abs(body["total_air_distance"]
               - s1["air_distance"] - s2["air_distance"]) < 1e-9


def test_cut_plan_three_level_nesting_inside_out(client):
    pid = _mk(client,
              "M0,0h100v100h-100Z M10,10h80v80h-80Z M40,40h20v20h-20Z")
    body = client.post(f"/paths/{pid}/cut-plan",
                       json={"x": 0, "y": 0}).json()
    assert [s["original_index"] for s in body["steps"]] == [2, 1, 0]
    # 链式空走：岛 (40,40) -> 孔 (10,10) -> 外环 (0,0)
    assert abs(body["steps"][0]["air_distance"]
               - (40 * 2 ** 0.5)) < 1e-9
    assert abs(body["steps"][1]["air_distance"]
               - (30 * 2 ** 0.5)) < 1e-9
    assert abs(body["steps"][2]["air_distance"]
               - (10 * 2 ** 0.5)) < 1e-9


def test_cut_plan_tie_breaks_by_original_index_then_vertex(client):
    # 两个独立外环，起点 (15,0)：到两轮廓最近顶点等距，原始索引小者先
    pid = _mk(client, "M0,0h10v10h-10Z M20,0h10v10h-10Z")
    body = client.post(f"/paths/{pid}/cut-plan",
                       json={"x": 15, "y": 0}).json()
    s1, s2 = body["steps"]
    assert s1["original_index"] == 0
    assert s1["entry_point"] == {"x": 10.0, "y": 0.0}
    assert s2["original_index"] == 1
    assert s2["entry_point"] == {"x": 20.0, "y": 0.0}
    assert body["total_air_distance"] == 15.0

    # 起点在正方形中心：四顶点等距，顶点序号 0 决胜
    pid2 = _mk(client, "M0,0h10v10h-10Z")
    body = client.post(f"/paths/{pid2}/cut-plan",
                       json={"x": 5, "y": 5}).json()
    assert body["steps"][0]["entry_vertex"] == 0
    assert body["steps"][0]["entry_point"] == {"x": 0.0, "y": 0.0}


def test_cut_plan_skips_open_and_degenerate(client):
    pid = _mk(client,
              "M0,0h10v10h-10Z M5,5 L8,5 L5,5 Z M20,0 L30,0 L30,10")
    body = client.post(f"/paths/{pid}/cut-plan",
                       json={"x": 0, "y": 0}).json()
    assert body["total_contours"] == 1
    assert [s["original_index"] for s in body["steps"]] == [0]
    reasons = {(s["original_index"], s["reason"])
               for s in body["skipped_contours"]}
    assert reasons == {(1, "degenerate"), (2, "open")}


def test_cut_plan_no_valid_contours_empty_plan(client):
    pid = _mk(client, "M0,0 L10,0 L10,10")
    body = client.post(f"/paths/{pid}/cut-plan",
                       json={"x": 3, "y": 4}).json()
    assert body["total_contours"] == 0
    assert body["steps"] == []
    assert body["total_air_distance"] == 0.0
    assert body["start"] == {"x": 3.0, "y": 4.0}
    assert body["skipped_contours"][0]["reason"] == "open"


def test_cut_plan_is_readonly_and_needs_no_prior_normalization(client):
    pid = _mk(client, "M0,0h20v20h-20Z M5,5h10v10h-10Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    assert client.post(f"/paths/{pid}/cut-plan",
                       json={"x": 0, "y": 0}).status_code == 200
    # 即时规划不写库
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    b1 = client.post(f"/paths/{pid}/cut-plan",
                     json={"x": 0, "y": 0}).json()
    b2 = client.post(f"/paths/{pid}/cut-plan",
                     json={"x": 0, "y": 0}).json()
    assert b1 == b2


def test_cut_plan_404(client):
    assert client.post("/paths/999999/cut-plan",
                       json={"x": 0, "y": 0}).status_code == 404


def test_cut_plan_422_on_invalid_start(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    url = f"/paths/{pid}/cut-plan"
    assert client.post(url, json={"y": 0}).status_code == 422       # 缺 x
    assert client.post(url, json={"x": 1}).status_code == 422       # 缺 y
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"x": "a", "y": 0}).status_code == 422
    assert client.post(url, json={"x": 0, "y": "nan"}).status_code == 422
    # NaN / Infinity / 超大数（非有限）均须 422 而不是 500
    assert client.post(url, json={"x": "nan", "y": 0}).status_code == 422
    r = client.post(url,
                    content='{"x": 1e999, "y": 0}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    r = client.post(url,
                    content='{"x": 0, "y": -1e999}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_cut_plan_422_on_unparseable_stored_path(client):
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    r = client.post(f"/paths/{bad_id}/cut-plan", json={"x": 0, "y": 0})
    assert r.status_code == 422
    assert "无法解析" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /paths/{id}/tabs 外环微桥布置（只读）
# ---------------------------------------------------------------------------

def test_tabs_simple_square(client):
    pid = _mk(client, "M0,0h40v40h-40Z")
    r = client.post(f"/paths/{pid}/tabs",
                    json={"count": 4, "width": 4, "corner_margin": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid
    assert body["fill_rule"] == "nonzero"
    assert body["count"] == 4
    assert body["width"] == 4.0
    assert body["corner_margin"] == 5.0
    assert body["contour_count"] == 1
    assert body["tab_count"] == 4
    assert body["unplaced_count"] == 0
    assert body["skipped_contours"] == []

    ce = body["contours"][0]
    assert ce["original_index"] == 0
    assert ce["role"] == "outer" and ce["depth"] == 0
    assert ce["perimeter"] == 160.0
    assert ce["requested_count"] == 4
    assert ce["tab_count"] == 4 and ce["unplaced_count"] == 0
    assert [t["sector"] for t in ce["tabs"]] == [0, 1, 2, 3]
    assert ce["tabs"][0] == {
        "sector": 0,
        "start": {"x": 18.0, "y": 0.0, "arc": 18.0},
        "end": {"x": 22.0, "y": 0.0, "arc": 22.0},
        "center": {"x": 20.0, "y": 0.0, "arc": 20.0}}


def test_tabs_default_corner_margin(client):
    pid = _mk(client, "M0,0h40v40h-40Z")
    r = client.post(f"/paths/{pid}/tabs", json={"count": 4, "width": 4})
    assert r.status_code == 200, r.text
    assert r.json()["corner_margin"] == 0.0


def test_tabs_skips_holes_open_and_degenerate(client):
    d = ("M0,0h40v40h-40Z M5,5h30v30h-30Z "     # outer + hole
         "M50,0 L60,0 L60,10 "                    # open
         "M20,20 L25,20 L20,20 Z")                # degenerate
    pid = _mk(client, d)
    body = client.post(f"/paths/{pid}/tabs",
                       json={"count": 1, "width": 2}).json()
    assert body["contour_count"] == 1
    assert body["contours"][0]["original_index"] == 0
    assert {(s["original_index"], s["reason"])
            for s in body["skipped_contours"]} == {
                (1, "hole"), (2, "open"), (3, "degenerate")}


def test_tabs_island_outer_at_depth_two_is_processed(client):
    pid = _mk(client,
              "M0,0h100v100h-100Z M10,10h80v80h-80Z M40,40h20v20h-20Z")
    body = client.post(f"/paths/{pid}/tabs",
                       json={"count": 2, "width": 1,
                             "corner_margin": 1}).json()
    assert [c["original_index"] for c in body["contours"]] == [0, 2]
    assert body["contours"][1]["depth"] == 2
    assert {(s["original_index"], s["reason"])
            for s in body["skipped_contours"]} == {(1, "hole")}


def test_tabs_reports_unplaced(client):
    # 边长 4 的正方形、桥宽 100：没有任何边放得下
    pid = _mk(client, "M0,0h4v4h-4Z")
    body = client.post(f"/paths/{pid}/tabs",
                       json={"count": 2, "width": 100}).json()
    ce = body["contours"][0]
    assert ce["tabs"] == []
    assert ce["tab_count"] == 0 and ce["unplaced_count"] == 2
    assert body["tab_count"] == 0 and body["unplaced_count"] == 2


def test_tabs_geometric_unfeasibility_is_not_a_422(client):
    # 几何放不下属于结果（unplaced_count），不是参数错误
    pid = _mk(client, "M0,0h4v4h-4Z")
    r = client.post(f"/paths/{pid}/tabs",
                    json={"count": 100, "width": 1000})
    assert r.status_code == 200
    assert r.json()["unplaced_count"] == 100


def test_tabs_no_valid_outer(client):
    pid = _mk(client, "M0,0 L10,0 L10,10")
    body = client.post(f"/paths/{pid}/tabs",
                       json={"count": 1, "width": 1}).json()
    assert body["contours"] == []
    assert body["contour_count"] == 0
    assert body["tab_count"] == 0
    assert body["skipped_contours"] == [
        {"original_index": 0, "role": "open", "reason": "open"}]


def test_tabs_is_readonly_and_needs_no_prior_normalization(client):
    pid = _mk(client, "M0,0h20v20h-20Z M5,5h10v10h-10Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    assert client.post(f"/paths/{pid}/tabs",
                       json={"count": 3, "width": 1}).status_code == 200
    # 即时计算不写库
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    b1 = client.post(f"/paths/{pid}/tabs",
                     json={"count": 3, "width": 1}).json()
    b2 = client.post(f"/paths/{pid}/tabs",
                     json={"count": 3, "width": 1}).json()
    assert b1 == b2


def test_tabs_404(client):
    assert client.post("/paths/999999/tabs",
                       json={"count": 1, "width": 1}).status_code == 404


def test_tabs_422_on_invalid_params(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    url = f"/paths/{pid}/tabs"
    assert client.post(url, json={"width": 1}).status_code == 422          # 缺 count
    assert client.post(url, json={"count": 2}).status_code == 422          # 缺 width
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"count": 0, "width": 1}).status_code == 422
    assert client.post(url, json={"count": -1, "width": 1}).status_code == 422
    assert client.post(url, json={"count": 2, "width": 0}).status_code == 422
    assert client.post(url, json={"count": 2, "width": -3}).status_code == 422
    assert client.post(url,
                       json={"count": 2, "width": 1,
                             "corner_margin": -1}).status_code == 422
    assert client.post(url,
                       json={"count": "x", "width": 1}).status_code == 422
    assert client.post(url,
                       json={"count": 2, "width": "nan"}).status_code == 422
    # 超过每轮廓微桥数上限 -> 422
    from app.tabs import MAX_TABS
    assert client.post(url, json={"count": MAX_TABS + 1,
                                  "width": 1}).status_code == 422
    # NaN / Infinity / 超大数（非有限）均须 422 而不是 500
    r = client.post(url,
                    content='{"count": 2, "width": 1e999}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    r = client.post(url,
                    content='{"count": 2, "width": 1, "corner_margin": 1e999}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_tabs_422_on_unparseable_stored_path(client):
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    r = client.post(f"/paths/{bad_id}/tabs",
                    json={"count": 1, "width": 1})
    assert r.status_code == 422
    assert "无法解析" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /paths/{id}/intersections 加工前轮廓交叉清单（只读）
# ---------------------------------------------------------------------------

def test_intersections_clean_path_empty(client):
    pid = _mk(client, "M0,0h40v40h-40Z M10,10h20v20h-20Z")
    r = client.post(f"/paths/{pid}/intersections")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid
    assert body["fill_rule"] == "nonzero"
    assert body["checked_contour_count"] == 2
    assert body["intersection_count"] == 0
    assert body["intersections"] == []
    assert body["skipped_contours"] == []


def test_intersections_empty_path(client):
    pid = _mk(client, "")
    r = client.post(f"/paths/{pid}/intersections")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["intersections"] == []
    assert body["intersection_count"] == 0
    assert body["checked_contour_count"] == 0


def test_intersections_self_crossing(client):
    # 蝴蝶结：边 0 与边 2 在 (5,5) 自交
    pid = _mk(client, "M0,0 L10,10 L10,0 L0,10 Z")
    body = client.post(f"/paths/{pid}/intersections").json()
    assert body["intersection_count"] == 1
    item = body["intersections"][0]
    assert item["type"] == "crossing"
    assert item["point"] == {"x": 5.0, "y": 5.0}
    assert item["contours"] == [
        {"original_index": 0, "role": "degenerate", "edge_indices": [0, 2]}]


def test_intersections_endpoint_contact_and_edge_indices(client):
    # 两个轮廓仅共享角点 (10,0)：端点接触，合并两侧边索引
    pid = _mk(client, "M0,0h10v10h-10Z M10,0 L20,0 L20,10 Z")
    body = client.post(f"/paths/{pid}/intersections").json()
    assert body["intersection_count"] == 1
    item = body["intersections"][0]
    assert item["type"] == "endpoint_contact"
    assert item["point"] == {"x": 10.0, "y": 0.0}
    assert item["contours"][0]["original_index"] == 0
    assert item["contours"][0]["edge_indices"] == [0, 1]
    assert item["contours"][1]["original_index"] == 1
    assert item["contours"][1]["edge_indices"] == [0, 2]


def test_intersections_collinear_overlap_gives_both_ends(client):
    # 第二轮廓顶边 (2,0)->(8,0) 与第一轮廓底边共线重叠
    pid = _mk(client, "M0,0h10v10h-10Z M2,-10 L8,-10 L8,0 L2,0 Z")
    body = client.post(f"/paths/{pid}/intersections").json()
    overlaps = [i for i in body["intersections"]
                if i["type"] == "collinear_overlap"]
    assert len(overlaps) == 1
    o = overlaps[0]
    assert o["points"] == [{"x": 2.0, "y": 0.0}, {"x": 8.0, "y": 0.0}]
    assert [g["original_index"] for g in o["contours"]] == [0, 1]
    # 重叠两端同时报告端点接触
    contacts = {(i["point"]["x"], i["point"]["y"])
                for i in body["intersections"]
                if i["type"] == "endpoint_contact"}
    assert contacts == {(2.0, 0.0), (8.0, 0.0)}


def test_intersections_same_point_merges_across_three_contours(client):
    pid = _mk(client,
              "M0,0h10v10h-10Z M10,0h10v10h-10Z M5,-5 L10,-5 L10,0 Z")
    body = client.post(f"/paths/{pid}/intersections").json()
    shared = [i for i in body["intersections"]
              if i["type"] == "endpoint_contact"
              and i["point"] == {"x": 10.0, "y": 0.0}]
    assert len(shared) == 1  # 同坐标同类型只出一条
    assert [g["original_index"] for g in shared[0]["contours"]] == [0, 1, 2]


def test_intersections_adjacent_edges_not_reported(client):
    pid = _mk(client, "M0,0 L10,0 L10,10 L0,10 Z")
    body = client.post(f"/paths/{pid}/intersections").json()
    assert body["intersections"] == []


def test_intersections_open_skipped(client):
    # 非退化开放折线穿过方块，仍不参与检查
    pid = _mk(client, "M0,0h10v10h-10Z M-5,5 L15,5 L15,6")
    body = client.post(f"/paths/{pid}/intersections").json()
    assert body["checked_contour_count"] == 1
    assert body["intersection_count"] == 0
    assert body["skipped_contours"] == [
        {"original_index": 1, "role": "open", "reason": "open"}]


def test_intersections_closed_degenerate_checked(client):
    pid = _mk(client, "M0,0 L10,0 L0,0 Z M5,0 L15,0 L15,10 Z")
    body = client.post(f"/paths/{pid}/intersections").json()
    assert body["checked_contour_count"] == 2
    assert body["skipped_contours"] == []
    assert body["intersection_count"] >= 1


def test_intersections_sorted_stably(client):
    d = ("M0,0 L0,10 L10,10 L10,0 Z M-5,2 L15,2 L15,8 L-5,8 Z")
    pid = _mk(client, d)
    bodies = [client.post(f"/paths/{pid}/intersections").json()
              for _ in range(3)]
    assert bodies[0] == bodies[1] == bodies[2]
    crossings = [(i["point"]["x"], i["point"]["y"])
                 for i in bodies[0]["intersections"] if i["type"] == "crossing"]
    assert crossings == [(0.0, 2.0), (0.0, 8.0), (10.0, 2.0), (10.0, 8.0)]


def test_intersections_is_readonly_and_needs_no_prior_normalization(client):
    pid = _mk(client, "M0,0 L10,10 L10,0 L0,10 Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    assert client.post(f"/paths/{pid}/intersections").status_code == 200
    # 即时检查不写库
    assert client.get(f"/paths/{pid}").json()["normalized"] is None


def test_intersections_404(client):
    assert client.post("/paths/999999/intersections").status_code == 404


def test_intersections_id_path_validation(client):
    assert client.post("/paths/x/intersections").status_code == 422
    assert client.post("/paths/0/intersections").status_code == 404


def test_intersections_422_on_unparseable_stored_path(client):
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    r = client.post(f"/paths/{bad_id}/intersections")
    assert r.status_code == 422
    assert "无法解析" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /paths/{id}/hit-test 批量点命中测试（只读）
# ---------------------------------------------------------------------------

def test_hit_test_inside_outside_boundary_order(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    r = client.post(f"/paths/{pid}/hit-test", json={"points": [
        {"x": 5, "y": 5}, {"x": -1, "y": 5}, {"x": 0, "y": 5},
        {"x": 10, "y": 10}, {"x": 5, "y": 0}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid
    assert body["fill_rule"] == "nonzero"
    assert body["point_count"] == 5
    assert body["checked_contour_count"] == 1
    assert body["skipped_contours"] == []
    assert [item["result"] for item in body["results"]] == [
        "inside", "outside", "boundary", "boundary", "boundary"]
    assert [item["index"] for item in body["results"]] == [0, 1, 2, 3, 4]
    assert body["results"][0]["point"] == {"x": 5.0, "y": 5.0}


def test_hit_test_uses_saved_fill_rule(client):
    # 两个同向方块重叠：nonzero 填充，evenodd 留空
    d = "M0,0h10v10h-10Z M0,0h10v10h-10Z"
    nz = _mk(client, d, fill_rule="nonzero")
    eo = _mk(client, d, fill_rule="evenodd")
    body = client.post(f"/paths/{nz}/hit-test",
                       json={"points": [{"x": 5, "y": 5}]}).json()
    assert body["fill_rule"] == "nonzero"
    assert body["results"][0]["result"] == "inside"
    body = client.post(f"/paths/{eo}/hit-test",
                       json={"points": [{"x": 5, "y": 5}]}).json()
    assert body["fill_rule"] == "evenodd"
    assert body["results"][0]["result"] == "outside"


def test_hit_test_nonzero_opposite_winding_cancels(client):
    # 外框顺时针 + 同几何逆时针：绕数抵消为 0
    pid = _mk(client,
              "M0,0h10v10h-10Z M0,0 L0,10 L10,10 L10,0 Z",
              fill_rule="nonzero")
    body = client.post(f"/paths/{pid}/hit-test",
                       json={"points": [{"x": 5, "y": 5}]}).json()
    assert body["results"][0]["result"] == "outside"


def test_hit_test_boundary_takes_priority(client):
    # 点同时在孔边（交叉为偶数本应 outside）上 -> 一律 boundary
    pid = _mk(client, "M0,0h10v10h-10Z M2,2h6v6h-6Z", fill_rule="evenodd")
    body = client.post(f"/paths/{pid}/hit-test", json={"points": [
        {"x": 5, "y": 2}, {"x": 2, "y": 2}, {"x": 5, "y": 5}]}).json()
    assert [i["result"] for i in body["results"]] == [
        "boundary", "boundary", "outside"]


def test_hit_test_open_subpaths_skipped(client):
    # 开放折线不参与：点落在开放折线上也只是 outside
    pid = _mk(client, "M0,0h10v10h-10Z M20,0 L30,0 L30,10")
    body = client.post(f"/paths/{pid}/hit-test", json={"points": [
        {"x": 5, "y": 5}, {"x": 25, "y": 0}]}).json()
    assert [i["result"] for i in body["results"]] == ["inside", "outside"]
    assert body["checked_contour_count"] == 1
    assert body["skipped_contours"] == [
        {"original_index": 1, "role": "open", "reason": "open"}]


def test_hit_test_zero_area_closed_self_intersection_participates(client):
    # 蝴蝶结（零净面积闭合自交）：叶内 inside、两叶之间 outside
    pid = _mk(client, "M0,0 L10,10 L10,0 L0,10 Z")
    body = client.post(f"/paths/{pid}/hit-test", json={"points": [
        {"x": 1, "y": 5}, {"x": 5, "y": 2}, {"x": 5, "y": 5}]}).json()
    assert [i["result"] for i in body["results"]] == [
        "inside", "outside", "boundary"]
    assert body["skipped_contours"] == []
    assert body["checked_contour_count"] == 1


def test_hit_test_curve_flattened(client):
    pid = _mk(client, "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z")
    body = client.post(f"/paths/{pid}/hit-test", json={"points": [
        {"x": 0, "y": 0}, {"x": -1, "y": 0}, {"x": 2, "y": 0}]}).json()
    assert [i["result"] for i in body["results"]] == [
        "inside", "boundary", "outside"]


def test_hit_test_empty_path_all_outside(client):
    pid = _mk(client, "")
    body = client.post(f"/paths/{pid}/hit-test",
                       json={"points": [{"x": 0, "y": 0}]}).json()
    assert body["results"][0]["result"] == "outside"
    assert body["checked_contour_count"] == 0


def test_hit_test_is_readonly_and_needs_no_prior_normalization(client):
    pid = _mk(client, "M0,0h10v10h-10Z M2,2h6v6h-6Z")
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    assert client.post(f"/paths/{pid}/hit-test",
                       json={"points": [{"x": 5, "y": 5}]}).status_code == 200
    # 即时计算不写库，重复调用一致
    assert client.get(f"/paths/{pid}").json()["normalized"] is None
    payload = {"points": [{"x": 1, "y": 1}, {"x": 5, "y": 5}]}
    b1 = client.post(f"/paths/{pid}/hit-test", json=payload).json()
    b2 = client.post(f"/paths/{pid}/hit-test", json=payload).json()
    assert b1 == b2


def test_hit_test_404(client):
    assert client.post("/paths/999999/hit-test",
                       json={"points": [{"x": 0, "y": 0}]}).status_code == 404


def test_hit_test_id_path_validation(client):
    assert client.post("/paths/x/hit-test",
                       json={"points": [{"x": 0, "y": 0}]}).status_code == 422
    assert client.post("/paths/0/hit-test",
                       json={"points": [{"x": 0, "y": 0}]}).status_code == 404


def test_hit_test_422_on_invalid_points(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    url = f"/paths/{pid}/hit-test"
    # 空点列表
    assert client.post(url, json={"points": []}).status_code == 422
    # 缺少 points
    assert client.post(url, json={}).status_code == 422
    # 点缺坐标 / 非数字
    assert client.post(url, json={"points": [{"x": 1}]}).status_code == 422
    assert client.post(url, json={"points": [{"x": "a", "y": 0}]}).status_code == 422
    assert client.post(url, json={"points": [{"x": 0, "y": "nan"}]}).status_code == 422
    # NaN / Infinity / 超大数（非有限）均须 422
    assert client.post(url,
                       content='{"points":[{"x":1e999,"y":0}]}',
                       headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post(url,
                       content='{"points":[{"x":0,"y":-1e999}]}',
                       headers={"Content-Type": "application/json"}).status_code == 422


def test_hit_test_422_when_over_500_points(client):
    pid = _mk(client, "M0,0h10v10h-10Z")
    url = f"/paths/{pid}/hit-test"
    pts = [{"x": 0, "y": 0}] * 500
    r = client.post(url, json={"points": pts})
    assert r.status_code == 200
    assert r.json()["point_count"] == 500
    pts501 = [{"x": 0, "y": 0}] * 501
    r = client.post(url, json={"points": pts501})
    assert r.status_code == 422


def test_hit_test_422_on_unparseable_stored_path(client):
    from app import db
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paths (d, fill_rule, metadata, contour_metadata) "
            "VALUES (?, 'nonzero', '{}', '[]')",
            ("M0,0 L",))
        bad_id = cur.lastrowid
    r = client.post(f"/paths/{bad_id}/hit-test",
                    json={"points": [{"x": 0, "y": 0}]})
    assert r.status_code == 422
    assert "无法解析" in r.json()["detail"]

