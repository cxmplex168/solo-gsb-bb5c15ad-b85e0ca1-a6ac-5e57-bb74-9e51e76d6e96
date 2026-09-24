"""孔洞最大内接圆选点（hole-clearance，只读即时分析），供钻孔选点。

对一份已存 SVG path 即时解析并完成轮廓包含层级分类后（与 metrics /
clearance / scanlines 等同一份解析与曲线细分折线，见
:func:`app.svggeo.analyze_contours`），为每个**有效孔洞**
（``role == "hole"``，闭合且非退化、包含深度为奇数）寻找孔内
**最大内接圆**的圆心：

- 候选圆心必须位于孔洞内部且不在任何**直属岛**（以该孔为直接
  parent 的外环，深度 = 孔深 + 1）内部；圆半径取圆心到孔洞边界与
  直属岛边界的**最短距离**（细分折线逐边的点-线段距离）；
- 搜索在孔洞包围盒上做**四叉树网格细分**：每个网格单元以「单元中心
  有符号净空 + 单元半对角线长」作为该单元可达半径的**距离上界**
  （有符号距离函数 1-Lipschitz），大根堆按上界优先细分最可能藏有
  更大圆的单元；水平/垂直扫描线的自由区间中点提供初始候选，避免
  凹孔或偏心岛使包围盒中心失效；
- 当堆中最大上界与当前最优半径之差不超过 ``precision`` 时收敛：
  返回半径与真实最大半径之差不超过 ``precision``；
- 全部孔洞合计最多处理 :data:`MAX_GRID_CELLS` 个网格单元，预算
  耗尽仍未达到精度时抛出 ``ValueError``（路由层返回 422）。

整个计算只读、即时完成：不依赖已保存的规范化结果，也不写库。
开放（``open``）与退化（``degenerate``）子路径跳过并列入
``skipped_contours``；没有有效孔洞时 ``holes`` 为空数组。
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any

from .svggeo import Contour, point_in_ring, point_segment_dist2

# 单次请求允许处理的网格单元总数上限（全部孔洞合计），超限由路由层返回 422
MAX_GRID_CELLS = 20000
# 输出量化位数（仅影响 JSON 展示，不改变判定）
_DECIMALS = 9
# 初始候选扫描线数量（水平/垂直各取该数量）
_SEED_LINES = 8

# 单元格：(minx, miny, maxx, maxy)
_Cell = tuple[float, float, float, float]


def compute_hole_clearance(contours: list[Contour],
                           precision: float,
                           max_cells: int = MAX_GRID_CELLS) -> dict[str, Any]:
    """对已分类的轮廓列表为每个有效孔洞寻找最大内接圆心。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓
            （已带 ``role`` / ``depth`` / ``parent_original_index``）。
        precision: 半径逼近精度，必须为大于 0 的有限数；返回半径与
            真实最大半径之差不超过该值。
        max_cells: 全部孔洞合计允许处理的网格单元数上限。

    返回:
        ``{precision, hole_count, cells_used, holes, skipped_contours}``；
        ``holes`` 按 ``original_index`` 升序，每项含
        ``{original_index, role, depth, center: {x, y}, radius,
        cells_used}``。

    抛出:
        ValueError: 参数非法，或网格单元预算耗尽仍未达到精度。
    """
    if not (isinstance(precision, (int, float))
            and math.isfinite(precision) and precision > 0):
        raise ValueError("precision 必须是大于 0 的有限数")
    if not isinstance(max_cells, int) or isinstance(max_cells, bool) \
            or max_cells <= 0:
        raise ValueError("max_cells 必须为正整数")
    precision = float(precision)

    ordered = sorted(contours, key=lambda c: c.original_index)
    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": ("open" if c.role == "open" else "degenerate")}
        for c in ordered if c.role in ("open", "degenerate")
    ]

    budget = _CellBudget(max_cells)
    holes: list[dict[str, Any]] = []
    for hole in [c for c in ordered if c.role == "hole"]:
        # 直属岛：以该孔为直接 parent 的外环（深度 = 孔深 + 1）
        islands = [c for c in ordered
                   if c.role == "outer"
                   and c.parent_original_index == hole.original_index]
        before = budget.used
        center, radius = _max_inscribed_circle(hole, islands, precision, budget)
        holes.append({
            "original_index": hole.original_index,
            "role": "hole",
            "depth": hole.depth,
            "center": {"x": _num(center[0]), "y": _num(center[1])},
            "radius": _num(radius),
            "cells_used": budget.used - before,
        })

    return {
        "precision": _num(precision),
        "hole_count": len(holes),
        "cells_used": budget.used,
        "holes": holes,
        "skipped_contours": skipped,
    }


# ---------------------------------------------------------------------------
# 单孔分支限界搜索
# ---------------------------------------------------------------------------

def _max_inscribed_circle(hole: Contour,
                          islands: list[Contour],
                          precision: float,
                          budget: "_CellBudget"
                          ) -> tuple[tuple[float, float], float]:
    """在单个孔洞的自由空间（孔内且直属岛外）内搜索最大内接圆心。

    返回 ``(center, radius)``；``radius`` 为 ``center`` 到孔洞/岛边界的
    最短距离，与真实最大半径之差不超过 ``precision``。
    """
    edges = _ring_edges(hole.points)
    for isl in islands:
        edges.extend(_ring_edges(isl.points))

    def signed_clearance(p: tuple[float, float]) -> tuple[float, bool]:
        """有符号净空：自由空间内为正（= 到边界最短距离），否则为负。"""
        d = math.sqrt(min(point_segment_dist2(p, a, b) for a, b in edges))
        return (d, True) if _in_free_space(p, hole, islands) else (-d, False)

    best_center: tuple[float, float] | None = None
    best_radius = 0.0

    def consider(p: tuple[float, float]) -> float:
        """评估一点并尝试更新最优；返回该点有符号净空。"""
        nonlocal best_center, best_radius
        s, valid = signed_clearance(p)
        if valid and s > best_radius:
            best_center, best_radius = p, s
        return s

    # 初始候选：扫描线自由区间中点（不依赖包围盒中心，
    # 凹孔或偏心岛导致包围盒中心落在自由空间之外时仍能提供孔内候选）
    for p in _seed_points(hole, islands):
        consider(p)

    # 分支限界：大根堆按「中心有符号净空 + 半对角线」的上界优先细分。
    # 初始网格把包围盒划分为接近正方形的起始单元（计入预算），
    # 避免细长包围盒在单一根单元上沿两轴无谓地层层全宽细分
    minx, miny, maxx, maxy = hole.bounds
    side = min(maxx - minx, maxy - miny)
    nx = max(1, math.ceil((maxx - minx) / side))
    ny = max(1, math.ceil((maxy - miny) / side))
    seq = itertools.count()  # 插入序号：上界相同按入堆先后决胜，保证确定性
    heap: list[tuple[float, int, _Cell]] = []
    budget.take(nx * ny)
    for iy in range(ny):
        for ix in range(nx):
            cell: _Cell = (minx + (maxx - minx) * ix / nx,
                           miny + (maxy - miny) * iy / ny,
                           minx + (maxx - minx) * (ix + 1) / nx,
                           miny + (maxy - miny) * (iy + 1) / ny)
            ub = consider(_center(cell)) + _half_diag(cell)
            if ub > best_radius:
                heapq.heappush(heap, (-ub, next(seq), cell))

    while heap:
        neg_ub, _, cell = heapq.heappop(heap)
        if -neg_ub <= best_radius + precision:
            break  # 堆顶即全局上界：其余单元上界均不更大，已达精度
        budget.take(4)
        child_hd = _half_diag(cell) / 2.0
        for child in _quadrants(cell):
            ub = consider(_center(child)) + child_hd
            if ub > best_radius:
                heapq.heappush(heap, (-ub, next(seq), child))
    # 堆空即收敛：不存在上界超过当前最优的未探索单元

    if best_center is None:
        # 任何候选半径都不超过 precision（自由空间薄到未被网格命中）
        # 的兜底：加密扫描线再找一次，仍无果退到包围盒中心、半径 0
        best_center = _last_resort_center(hole, islands)
        best_radius = 0.0
    return best_center, best_radius


class _CellBudget:
    """全部孔洞共享的网格单元计数预算，耗尽即抛 ``ValueError``。"""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def take(self, n: int) -> None:
        if self.used + n > self.limit:
            raise ValueError(
                f"网格单元达到上限 {self.limit} 仍未达到精度，请增大 precision")
        self.used += n


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def _ring_edges(points: list[tuple[float, float]]
                ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """闭合环的边序列（含末点回首点的闭合边）。"""
    n = len(points)
    return [(points[k], points[(k + 1) % n]) for k in range(n)]


def _in_free_space(p: tuple[float, float],
                   hole: Contour,
                   islands: list[Contour]) -> bool:
    """点是否位于孔洞自由空间：严格在孔内且不在任何直属岛内部。"""
    if point_in_ring(p, hole.points) != 1:
        return False
    return all(point_in_ring(p, isl.points) != 1 for isl in islands)


def _center(cell: _Cell) -> tuple[float, float]:
    return ((cell[0] + cell[2]) / 2.0, (cell[1] + cell[3]) / 2.0)


def _half_diag(cell: _Cell) -> float:
    return math.hypot(cell[2] - cell[0], cell[3] - cell[1]) / 2.0


def _quadrants(cell: _Cell) -> list[_Cell]:
    """四等分单元（固定顺序，保证细分过程确定）。"""
    x0, y0, x1, y1 = cell
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return [
        (x0, y0, mx, my),
        (mx, y0, x1, my),
        (x0, my, mx, y1),
        (mx, my, x1, y1),
    ]


# ---------------------------------------------------------------------------
# 初始候选：扫描线自由区间中点
# ---------------------------------------------------------------------------

def _seed_points(hole: Contour,
                 islands: list[Contour],
                 line_count: int = _SEED_LINES
                 ) -> list[tuple[float, float]]:
    """水平/垂直扫描线上自由区间的中点，作为分支限界的初始候选。"""
    minx, miny, maxx, maxy = hole.bounds
    width, height = maxx - minx, maxy - miny
    points: list[tuple[float, float]] = []
    for k in range(line_count):
        x = minx + width * (k + 0.5) / line_count
        points.extend(_free_midpoints(hole, islands, axis="x", const=x))
        y = miny + height * (k + 0.5) / line_count
        points.extend(_free_midpoints(hole, islands, axis="y", const=y))
    return points


def _free_midpoints(hole: Contour,
                    islands: list[Contour],
                    *,
                    axis: str,
                    const: float) -> list[tuple[float, float]]:
    """直线 ``axis=const`` 上「孔内区间 − 岛区间」各自由段的中点。"""
    hole_iv = _intervals_on_line(hole.points, const, axis)
    blocks: list[tuple[float, float]] = []
    for isl in islands:
        blocks.extend(_intervals_on_line(isl.points, const, axis))
    mids: list[tuple[float, float]] = []
    for a, b in _subtract_intervals(hole_iv, blocks):
        mid = (a + b) / 2.0
        mids.append((const, mid) if axis == "x" else (mid, const))
    return mids


def _intervals_on_line(points: list[tuple[float, float]],
                       const: float,
                       axis: str) -> list[tuple[float, float]]:
    """半开边规则下闭合环与直线的交点两两配对成的区间（沿另一轴）。

    与扫描线同一口径 ``(c1 > const) != (c2 > const)``：垂直于 axis 的
    边永不计数，穿过顶点时只被两条邻边之一计数。
    """
    vals: list[float] = []
    n = len(points)
    for k in range(n):
        a, b = points[k], points[(k + 1) % n]
        ca, cb = (a[0], b[0]) if axis == "x" else (a[1], b[1])
        oa, ob = (a[1], b[1]) if axis == "x" else (a[0], b[0])
        if (ca > const) != (cb > const):
            vals.append(oa + (ob - oa) * (const - ca) / (cb - ca))
    vals.sort()
    if len(vals) % 2 == 1:
        vals.pop()  # 异常相切配置防御性丢弃落单交点
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


def _subtract_intervals(base: list[tuple[float, float]],
                        blocks: list[tuple[float, float]]
                        ) -> list[tuple[float, float]]:
    """从升序互不重叠的 ``base`` 区间中扣除 ``blocks`` 区间的并。"""
    out: list[tuple[float, float]] = []
    for lo, hi in base:
        cur = lo
        for a, b in sorted(blocks):
            if b <= cur:
                continue
            if a >= hi:
                break
            if a > cur:
                out.append((cur, a))
            cur = max(cur, b)
            if cur >= hi:
                break
        if cur < hi:
            out.append((cur, hi))
    return [(a, b) for a, b in out if b - a > 0.0]


def _last_resort_center(hole: Contour,
                        islands: list[Contour]) -> tuple[float, float]:
    """兜底选点：加密扫描线找自由区间中点，仍无果退到包围盒中心。"""
    for p in _seed_points(hole, islands, line_count=64):
        if _in_free_space(p, hole, islands):
            return p
    minx, miny, maxx, maxy = hole.bounds
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)


def _num(v: float) -> float:
    """量化展示值并消除 -0.0。"""
    r = round(v, _DECIMALS)
    return r + 0.0 if r == 0 else r
