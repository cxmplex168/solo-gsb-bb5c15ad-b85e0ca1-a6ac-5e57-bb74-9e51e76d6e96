"""孔洞最大内接圆选点（hole-clearance）的几何单元测试。"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.holeclearance import (  # noqa: E402
    MAX_GRID_CELLS,
    compute_hole_clearance,
)
from app.svggeo import (  # noqa: E402
    analyze_contours,
    point_in_ring,
    point_segment_dist2,
)


def _run(d, precision=0.001, fill_rule="nonzero", max_cells=MAX_GRID_CELLS):
    contours, _ = analyze_contours(d, fill_rule)
    return compute_hole_clearance(contours, precision, max_cells=max_cells)


def _hole(res, original_index):
    return next(h for h in res["holes"] if h["original_index"] == original_index)


def _assert_entry_consistent(d, entry, fill_rule="nonzero"):
    """独立复核：圆心在孔内且避开直属岛，半径等于到边界的最短距离。"""
    contours, _ = analyze_contours(d, fill_rule)
    hole = next(c for c in contours
                if c.original_index == entry["original_index"])
    islands = [c for c in contours if c.role == "outer"
               and c.parent_original_index == hole.original_index]
    p = (entry["center"]["x"], entry["center"]["y"])
    assert point_in_ring(p, hole.points) == 1
    for isl in islands:
        assert point_in_ring(p, isl.points) != 1
    best = float("inf")
    for ring in [hole, *islands]:
        pts = ring.points
        for k in range(len(pts)):
            best = min(best, point_segment_dist2(p, pts[k], pts[(k + 1) % len(pts)]))
    assert abs(math.sqrt(best) - entry["radius"]) < 1e-9


# ---------------------------------------------------------------------------
# 基本形状
# ---------------------------------------------------------------------------

def test_square_hole_exact_center_and_radius():
    # 外方 0..20，内孔 4..16：最大内接圆半径 6，圆心恰为孔中心 (10,10)
    res = _run("M0,0h20v20h-20Z M4,4h12v12h-12Z")
    assert res["hole_count"] == 1
    assert res["precision"] == 0.001
    assert res["skipped_contours"] == []
    h = res["holes"][0]
    assert h["original_index"] == 1
    assert h["role"] == "hole" and h["depth"] == 1
    assert math.isclose(h["radius"], 6.0, abs_tol=1e-9)
    assert math.isclose(h["center"]["x"], 10.0, abs_tol=1e-9)
    assert math.isclose(h["center"]["y"], 10.0, abs_tol=1e-9)
    assert h["cells_used"] > 0
    assert res["cells_used"] == h["cells_used"]


def test_radius_within_precision_of_true_max():
    # 方孔 + 更紧的精度：返回半径与真实最大半径之差不超过 precision
    res = _run("M0,0h20v20h-20Z M4,4h12v12h-12Z", precision=1e-6)
    h = res["holes"][0]
    assert 6.0 - 1e-6 - 1e-9 <= h["radius"] <= 6.0 + 1e-9


def test_eccentric_island_defeats_bbox_center():
    # 孔 0..20，偏心岛 2..4：最优圆心被推到 (16√2-12, 16√2-12)，
    # 半径 32-16√2 ≈ 9.3726；包围盒中心 (10,10) 的净空只有 6√2 ≈ 8.485
    d = "M-10,-10h40v40h-40Z M0,0h20v20h-20Z M2,2h2v2h-2Z"
    res = _run(d)
    h = res["holes"][0]
    true_max = 32 - 16 * math.sqrt(2)
    assert h["radius"] > 8.5  # 明确优于包围盒中心
    assert abs(h["radius"] - true_max) <= 0.001 + 1e-9
    t = 16 * math.sqrt(2) - 12
    assert abs(h["center"]["x"] - t) < 0.01
    assert abs(h["center"]["y"] - t) < 0.01
    _assert_entry_consistent(d, h)


def test_concave_hole_avoids_bbox_center():
    # L 形凹孔：包围盒中心 (5,5) 落在自由空间之外；
    # 最优为肘部内切圆，半径 8-4√2 ≈ 2.3431
    d = "M-5,-5h30v30h-30Z M0,0 L10,0 L10,4 L4,4 L4,10 L0,10 Z"
    res = _run(d)
    h = res["holes"][0]
    true_max = 8 - 4 * math.sqrt(2)
    assert abs(h["radius"] - true_max) <= 0.001 + 1e-9
    _assert_entry_consistent(d, h)


def test_centered_island_four_symmetric_optima():
    # 孔 2..38（在外方 0..40 内），居中岛 18..22：
    # 四个角域对称最优，半径 (16√2)/(1+√2)·... = 角到岛角距离平衡值
    d = "M0,0h40v40h-40Z M2,2h36v36h-36Z M18,18h4v4h-4Z"
    res = _run(d)
    h = res["holes"][0]
    # 沿对角线：min(t, √2(16-t)) -> t = 16√2/(1+√2) = 32-16√2 ≈ 9.3726
    true_max = 32 - 16 * math.sqrt(2)
    assert abs(h["radius"] - true_max) <= 0.001 + 1e-9
    _assert_entry_consistent(d, h)


def test_nested_hole_in_island_processed_separately():
    # 外-孔(10..90)-岛(40..60)-孔(45..55)：两个孔各自求解
    d = ("M0,0h100v100h-100Z M10,10h80v80h-80Z "
         "M40,40h20v20h-20Z M45,45h10v10h-10Z")
    res = _run(d)
    assert [h["original_index"] for h in res["holes"]] == [1, 3]
    h1 = _hole(res, 1)
    # 孔 1 的直属岛是轮廓 2：角域最优 min(t-10, √2(40-t)) -> 60-30√2
    assert abs(h1["radius"] - (60 - 30 * math.sqrt(2))) <= 0.001 + 1e-9
    h3 = _hole(res, 3)
    # 孔 3 在岛内部，无直属岛：整个 10x10 方孔，半径 5
    assert math.isclose(h3["radius"], 5.0, abs_tol=1e-9)
    assert math.isclose(h3["center"]["x"], 50.0, abs_tol=1e-9)
    assert math.isclose(h3["center"]["y"], 50.0, abs_tol=1e-9)
    _assert_entry_consistent(d, h1)
    _assert_entry_consistent(d, h3)


def test_multiple_sibling_holes_sorted_by_index():
    d = "M0,0h100v100h-100Z M5,5h10v10h-10Z M80,80h10v10h-10Z"
    res = _run(d)
    assert [h["original_index"] for h in res["holes"]] == [1, 2]
    for h, (cx, cy) in zip(res["holes"], [(10.0, 10.0), (85.0, 85.0)]):
        assert math.isclose(h["radius"], 5.0, abs_tol=1e-9)
        assert math.isclose(h["center"]["x"], cx, abs_tol=1e-9)
        assert math.isclose(h["center"]["y"], cy, abs_tol=1e-9)
    assert res["cells_used"] == sum(h["cells_used"] for h in res["holes"])


def test_curve_hole_flattened():
    # 孔为单位圆（两段弧）：细分折线内接于圆，最大内接圆半径略小于 1
    d = "M-2,-2h4v4h-4Z M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    res = _run(d)
    h = res["holes"][0]
    assert 0.99 < h["radius"] < 1.0
    assert abs(h["center"]["x"]) < 0.01
    assert abs(h["center"]["y"]) < 0.01
    _assert_entry_consistent(d, h)


# ---------------------------------------------------------------------------
# 跳过规则与空结果
# ---------------------------------------------------------------------------

def test_no_valid_holes_returns_empty_array():
    for d in ("M0,0h10v10h-10Z", "", "M0,0 L10,0 L10,10"):
        res = _run(d)
        assert res["holes"] == []
        assert res["hole_count"] == 0
        assert res["cells_used"] == 0


def test_open_and_degenerate_skipped():
    d = ("M0,0h40v40h-40Z M5,5h30v30h-30Z "   # outer + hole
         "M41,0 L45,0 L45,5 "                  # open
         "M10,10 L15,10 L10,10 Z")             # degenerate
    res = _run(d)
    assert res["hole_count"] == 1
    assert math.isclose(res["holes"][0]["radius"], 15.0, abs_tol=1e-9)
    assert res["skipped_contours"] == [
        {"original_index": 2, "role": "open", "reason": "open"},
        {"original_index": 3, "role": "degenerate", "reason": "degenerate"},
    ]


def test_only_direct_islands_constrain():
    # 孔 1 的直属岛只有轮廓 2；岛内的孔 3 不影响孔 1 的自由空间
    d = ("M0,0h100v100h-100Z M10,10h80v80h-80Z "
         "M40,40h20v20h-20Z M48,48h4v4h-4Z")
    res = _run(d)
    h1 = _hole(res, 1)
    assert abs(h1["radius"] - (60 - 30 * math.sqrt(2))) <= 0.001 + 1e-9
    _assert_entry_consistent(d, h1)


def test_fill_rule_independent():
    d = "M0,0h20v20h-20Z M4,4h12v12h-12Z"
    nz = _run(d, fill_rule="nonzero")
    eo = _run(d, fill_rule="evenodd")
    assert nz["holes"] == eo["holes"]


# ---------------------------------------------------------------------------
# 网格预算与参数校验
# ---------------------------------------------------------------------------

def test_budget_exceeded_raises():
    # 长走廊孔 0..1000 x 0..10：中心线处处半径 5，
    # precision 1e-6 需要把整条走廊细分到亚微米级 -> 预算耗尽
    d = "M-10,-10 L1020,-10 L1020,30 L-10,30 Z M0,0 L1000,0 L1000,10 L0,10 Z"
    contours, _ = analyze_contours(d, "nonzero")
    with pytest.raises(ValueError, match="200"):
        compute_hole_clearance(contours, 1e-6, max_cells=200)


def test_budget_shared_across_holes():
    # 两个相同的走廊孔共享 20000 总预算：合计超限即抛错
    d = ("M-10,-10 L1020,-10 L1020,60 L-10,60 Z "
         "M0,0 L1000,0 L1000,10 L0,10 Z "
         "M0,40 L1000,40 L1000,50 L0,50 Z")
    contours, _ = analyze_contours(d, "nonzero")
    with pytest.raises(ValueError):
        compute_hole_clearance(contours, 1e-6, max_cells=20000)
    # 放宽精度后同一形状正常收敛
    res = compute_hole_clearance(contours, 0.5, max_cells=20000)
    assert res["hole_count"] == 2


def test_corridor_converges_with_loose_precision():
    # 同一走廊：精度 0.5 时在预算内收敛到半径 5（中心线平台）
    d = "M-10,-10 L1020,-10 L1020,30 L-10,30 Z M0,0 L1000,0 L1000,10 L0,10 Z"
    res = _run(d, precision=0.5)
    h = res["holes"][0]
    assert abs(h["radius"] - 5.0) <= 0.5 + 1e-9
    _assert_entry_consistent(d, h)


def test_invalid_precision_raises():
    contours, _ = analyze_contours("M0,0h20v20h-20Z M4,4h12v12h-12Z", "nonzero")
    for bad in (0, -1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            compute_hole_clearance(contours, bad)
    with pytest.raises(ValueError):
        compute_hole_clearance(contours, 0.1, max_cells=0)


def test_results_deterministic():
    d = "M-10,-10h40v40h-40Z M0,0h20v20h-20Z M2,2h2v2h-2Z"
    outs = [_run(d) for _ in range(3)]
    assert outs[0] == outs[1] == outs[2]
