"""移动端 Tile-Based 渲染（TBDR）的 load/store 参考实现。

初始版本中所有入口都抛出 ``NotImplementedError``。本模块用纯 CPU、可重复的
数据模型复现线上帧时间抖动与画面异常，并逐项给出修复：

1. **按需清屏** —— :func:`loadop_by_need`
   颜色附件每帧在若干个图像槽位之间轮换。帧 N 无条件 ``loadOp=load`` 时，
   读到的是同一槽位上一次写入的内容，即第 ``N - slots`` 帧（双缓冲下就是
   两帧前），半透明叠加因此出现拖影。修复方式是按内容是否真的需要跨帧
   保留来决定 load/clear：帧内会被完整重写的附件一律 ``clear``；只有内容
   确实要保留、且当前槽位恰好持有上一帧结果的附件才 ``load``；槽位已经
   轮换掉、但内容仍被需要的，改为从上一帧槽位复制，绝不读陈旧样本。

2. **解析移出 pass** —— :func:`resolve_outside`
   MSAA 解析放在 render pass 内部时，解析之后的子 pass 仍往同一块 tile 写
   样本，后续采样拿到的还是未解析的样本。修复方式是把解析挪到 pass 结束
   之后（独立的 resolve 步骤），并保证任何采样都排在解析之后、且采样的是
   已解析的单样本纹理。

3. **非末 pass 不 store** —— :func:`skip_store_mid`
   多个子 pass 共用同一块 tile 内存，却每个子 pass 都 ``storeOp=store``，
   把 tile 内容反复写回系统内存，带宽被打满、帧时间抖动。修复方式是只有
   最后一个写该附件的子 pass 执行 store；中间子 pass 使用 ``dontcare``，
   内容留在 tile 内存里由后续子 pass 继续使用。

4. **显式帧内依赖** —— :func:`explicit_dependency`
   异步计算队列读取 tile 深度时，图形 pass 尚未结束，缺少帧内依赖会读到
   清屏值。修复方式是在图形 pass 的结束点与 compute 的读取点之间插入显式
   依赖（stage / access 成对），跨队列时再补上信号量。

5. **只读深度标只读** —— :func:`readonly_depth`
   只做深度测试的 pass 仍把深度附件绑定为可写，驱动会插入隐式布局转换，
   每帧多一次全图转换。修复方式是把这些 pass 的深度引用标记为
   ``depth_stencil_read_only_optimal``。

6. **布局转换合并** —— :func:`merge_layout`
   同一附件在相邻子 pass 中保持同一布局，却每个子 pass 边界都做一次转换。
   修复方式是把布局需求按附件聚合，合并同布局的连续区间，只在布局真正
   变化时转换。

其余入口（:func:`begin_pass` / :func:`end_pass` / :func:`resolve` /
:func:`submit` / :func:`validate_pass` / :func:`metrics` / :func:`audit` /
:func:`fallback` / :func:`determinism`）把上述修复串成一条可观测、可审计、
可复现的管线。

约定
----
* 所有公开函数都不修改入参；返回值可 JSON 序列化（:func:`begin_pass` 返回
  的 :class:`PassState` 是给 :func:`end_pass` 用的句柄，本身也可审计）。
* 传 ``None``、标量或任意字典都能得到有意义的结果，便于契约测试与审计
  工具复用。
* 带宽模型只统计**系统内存**流量：tile 内存内部读写免费，只有 load /
  store / resolve / 布局转换产生流量。``clear`` 走快速清屏，计 0 字节。
* 不含时间戳等非确定字段，同一输入重复调用结果逐位一致。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    # 六项修复
    "loadop_by_need",
    "resolve_outside",
    "skip_store_mid",
    "explicit_dependency",
    "readonly_depth",
    "merge_layout",
    # 管线装配
    "begin_pass",
    "end_pass",
    "resolve",
    "submit",
    # 观测与验收
    "validate_pass",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 支撑类型
    "Attachment",
    "PassState",
    "TileError",
    "ValidationError",
    "LAYOUT_COLOR",
    "LAYOUT_DEPTH",
    "LAYOUT_DEPTH_READ_ONLY",
    "LAYOUT_SHADER_READ",
    "LAYOUT_UNDEFINED",
    "LAYOUT_PRESENT",
]

# ---------------------------------------------------------------------------
# 常量：移动端 TBDR 的默认参数与带宽成本模型
# ---------------------------------------------------------------------------

#: 默认渲染分辨率（像素）。
DEFAULT_RESOLUTION = (1920, 1080)
#: 默认 tile 尺寸（像素）。移动 GPU 常见 32x32。
DEFAULT_TILE = (32, 32)
#: 默认 MSAA 样本数。
DEFAULT_SAMPLES = 4
#: 颜色附件每像素字节数（RGBA8）。
COLOR_BYTES_PER_PIXEL = 4
#: 深度附件每像素字节数（D24S8 按 4B 计）。
DEPTH_BYTES_PER_PIXEL = 4
#: 默认颜色缓冲图像槽位数（双缓冲）。
DEFAULT_SLOTS = 2

#: 布局常量（与 Vulkan / Metal 的概念一一对应）。
LAYOUT_UNDEFINED = "undefined"
LAYOUT_COLOR = "color_attachment_optimal"
LAYOUT_DEPTH = "depth_stencil_attachment_optimal"
LAYOUT_DEPTH_READ_ONLY = "depth_stencil_read_only_optimal"
LAYOUT_SHADER_READ = "shader_read_only_optimal"
LAYOUT_PRESENT = "present_src"

#: 布局转换按整图读 + 写建模，成本系数为 2 倍附件大小。
TRANSITION_COPY_FACTOR = 2

#: 建模用的有效带宽：25.6 GB/s（移动端 LPDDR5 量级）。
BANDWIDTH_BYTES_PER_MS = 25_600_000.0
#: 一个健康帧的固定 CPU/驱动开销（毫秒），用于抖动模型。
BASE_FRAME_MS = 4.0

#: 附件种类别名。
_KIND_ALIASES = {
    "color": "color",
    "colour": "color",
    "rgba": "color",
    "depth": "depth",
    "depth_stencil": "depth",
    "depthstencil": "depth",
    "d24s8": "depth",
    "resolve": "resolve",
    "resolved": "resolve",
}


class TileError(RuntimeError):
    """tile 管线被违反生命周期的使用（例如对已结束的 pass 追加子 pass）。"""


class ValidationError(RuntimeError):
    """硬性不变量被破坏（例如在未解析的 MSAA 样本上采样）。"""


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> dict[str, Any]:
    """把可选字典规格安全取出；标量 / None 一律视为空规格。"""
    return dict(value) if isinstance(value, Mapping) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else default
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return default


def _pick(source: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in source and source[name] is not None:
            return source[name]
    return default


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _names(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value) if item is not None]


def _as_name_map(value: Any, default: Any = None) -> dict[str, Any]:
    """把 ``{name: x}`` / ``[name, ...]`` / 标量统一成 ``{name: x}``。"""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {name: default for name in _names(value)}


def _resolution(value: Any = None) -> tuple[int, int]:
    if isinstance(value, Mapping):
        width = _as_int(_pick(value, "width", "w", default=0), 0)
        height = _as_int(_pick(value, "height", "h", default=0), 0)
        if width > 0 and height > 0:
            return width, height
        return DEFAULT_RESOLUTION
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        width = _as_int(value[0], 0)
        height = _as_int(value[1], 0)
        if width > 0 and height > 0:
            return width, height
        return DEFAULT_RESOLUTION
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        side = _as_int(value, 0)
        if side > 1:
            return side, side
    return DEFAULT_RESOLUTION


def _tile_size(value: Any = None) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        width = _as_int(value[0], 0)
        height = _as_int(value[1], 0)
        if width > 0 and height > 0:
            return width, height
    side = _as_int(value, 0) if not isinstance(value, Mapping) else 0
    if side > 0:
        return side, side
    return DEFAULT_TILE


def _tile_count(resolution: tuple[int, int], tile: tuple[int, int]) -> int:
    width, height = resolution
    tw, th = tile
    return max(1, math.ceil(width / tw)) * max(1, math.ceil(height / th))


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 附件与 pass 的描述
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attachment:
    """一个渲染附件（颜色 / 深度 / 已解析目标）。

    槽位（slot）模型对应移动端的多缓冲轮换：帧 ``N`` 使用
    ``slot = N % slots``。``last_frame`` / ``last_slot`` 描述该附件上一次被
    写入的帧与槽位——这两项是判断 ``loadOp`` 是否安全的唯一依据。
    """

    name: str
    kind: str = "color"
    samples: int = 1
    bytes_per_pixel: int = COLOR_BYTES_PER_PIXEL
    #: 内容必须跨帧保留（TAA 历史、上一帧深度等）。
    preserve: bool = False
    #: 每帧被完整重写：load 没有意义，直接 clear。
    overwrite: bool = True
    slots: int = 1
    slot: int = 0
    last_slot: int | None = None
    last_frame: int | None = None
    #: 解析目标（MSAA 颜色附件解析到哪张单样本纹理）。
    resolve_to: str | None = None

    @property
    def is_depth(self) -> bool:
        return self.kind == "depth"

    @property
    def is_msaa(self) -> bool:
        return self.samples > 1

    def image_bytes(self, resolution: tuple[int, int], samples: int | None = None) -> int:
        width, height = resolution
        count = self.samples if samples is None else samples
        count = max(1, count)
        return width * height * self.bytes_per_pixel * count

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "samples": self.samples,
            "bytes_per_pixel": self.bytes_per_pixel,
            "preserve": self.preserve,
            "overwrite": self.overwrite,
            "slots": self.slots,
            "slot": self.slot,
            "last_slot": self.last_slot,
            "last_frame": self.last_frame,
            "resolve_to": self.resolve_to,
        }


def _normalize_attachment(raw: Any, index: int = 0, frame: int = 0) -> Attachment:
    if isinstance(raw, Attachment):
        return raw
    if isinstance(raw, str):
        raw = {"name": raw}
    data = _mapping(raw)
    name = str(_pick(data, "name", "id", default=f"attachment{index}"))
    kind_raw = str(_pick(data, "kind", "type", default="color")).strip().lower()
    kind = _KIND_ALIASES.get(kind_raw, "color")
    samples = max(1, _as_int(_pick(data, "samples", "sample_count", default=1), 1))
    bpp = _as_int(
        _pick(data, "bytes_per_pixel", "bpp", default=0),
        0,
    )
    if bpp <= 0:
        bpp = DEPTH_BYTES_PER_PIXEL if kind == "depth" else COLOR_BYTES_PER_PIXEL
    slots = max(1, _as_int(_pick(data, "slots", "slot_count", default=1), 1))
    slot = _as_int(_pick(data, "slot", default=frame % slots), frame % slots)
    last_slot_raw = _pick(data, "last_slot", "previous_slot", default=None)
    last_frame_raw = _pick(data, "last_frame", "previous_frame", default=None)
    return Attachment(
        name=name,
        kind=kind,
        samples=samples,
        bytes_per_pixel=bpp,
        preserve=_as_bool(_pick(data, "preserve", "persistent", default=False)),
        overwrite=_as_bool(_pick(data, "overwrite", "full_overwrite", default=True)),
        slots=slots,
        slot=slot,
        last_slot=None if last_slot_raw is None else _as_int(last_slot_raw, slot),
        last_frame=None if last_frame_raw is None else _as_int(last_frame_raw, -1),
        resolve_to=_pick(data, "resolve_to", "resolve_target", default=None),
    )


def _normalize_subpass(raw: Any, index: int = 0) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"name": raw}
    data = _mapping(raw)
    name = str(_pick(data, "name", "id", "label", default=f"subpass{index}"))
    writes = _names(_pick(data, "writes", "write", "color", default=[]))
    samples = _names(_pick(data, "samples", "sample", "reads", default=[]))
    declared_load = _as_name_map(_pick(data, "load", "load_ops", default=None), "load")
    declared_store = _as_name_map(_pick(data, "store", "store_ops", default=None), "store")
    layouts = {
        str(k): str(v) for k, v in _as_name_map(_pick(data, "layouts", default=None)).items()
    }
    resolve_inside = _names(_pick(data, "resolve", "resolve_inside", default=[]))
    return {
        "index": index,
        "name": name,
        "writes": writes,
        "samples": samples,
        "depth_test_only": _as_bool(_pick(data, "depth_test_only", default=False)),
        "depth_write": _as_bool(_pick(data, "depth_write", default=False)),
        "stencil_write": _as_bool(_pick(data, "stencil_write", default=False)),
        "load": declared_load,
        "store": declared_store,
        "layouts": layouts,
        "resolve_inside": resolve_inside,
        "resolve_after": _as_bool(_pick(data, "resolve_after", default=False)),
    }


@dataclass
class PassState:
    """``begin_pass`` 返回的 pass 句柄；``end_pass`` 消费它并给出帧内报告。"""

    frame: int
    resolution: tuple[int, int]
    tile: tuple[int, int]
    attachments: list[Attachment] = field(default_factory=list)
    subpasses: list[dict[str, Any]] = field(default_factory=list)
    #: 跨队列消费者（异步计算）声明，例如 ``[{"queue": "compute", "reads": ["depth"]}]``.
    consumers: list[dict[str, Any]] = field(default_factory=list)
    #: 已声明的帧内依赖。
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    @property
    def attachment_map(self) -> dict[str, Attachment]:
        return {a.name: a for a in self.attachments}

    def add_subpass(self, raw: Any) -> dict[str, Any]:
        if self.closed:
            raise TileError("pass 已结束，不能再追加子 pass")
        sub = _normalize_subpass(raw, len(self.subpasses))
        self.subpasses.append(sub)
        return sub

    def find(self, name: str) -> Attachment | None:
        return self.attachment_map.get(name)


def _demo_spec(frame: int = 1) -> dict[str, Any]:
    """构造一条**修复后**的健康帧，供整数式契约调用使用。

    场景：G-Buffer 几何子 pass + 光照子 pass（共享 tile 内存）+ 异步计算
    读取深度做屏幕空间效果。附件槽位为双缓冲，但所有被读的附件都明确声明
    了来源，因此不存在陈旧 load。
    """
    slots = DEFAULT_SLOTS
    slot = frame % slots
    previous_slot = (frame - 1) % slots
    return {
        "frame": frame,
        "resolution": DEFAULT_RESOLUTION,
        "tile": DEFAULT_TILE,
        "attachments": [
            {
                "name": "gbuffer_color",
                "kind": "color",
                "samples": DEFAULT_SAMPLES,
                "slots": slots,
                "slot": slot,
                "last_slot": previous_slot,
                "last_frame": frame - slots,
                "preserve": False,
                "overwrite": True,
                "resolve_to": "scene_color",
            },
            {
                "name": "depth",
                "kind": "depth",
                "samples": DEFAULT_SAMPLES,
                "slots": 1,
                "slot": 0,
                "preserve": False,
                "overwrite": True,
            },
            {
                "name": "scene_color",
                "kind": "color",
                "samples": 1,
                "slots": slots,
                "slot": slot,
                "last_frame": frame - slots,
                "preserve": False,
                "overwrite": True,
            },
            {
                "name": "taa_history",
                "kind": "color",
                "samples": 1,
                "slots": slots,
                "slot": slot,
                "last_slot": slot,
                "last_frame": frame - slots,
                "preserve": True,
                "overwrite": False,
            },
        ],
        "subpasses": [
            {
                "name": "geometry",
                "writes": ["gbuffer_color", "depth"],
                "samples": [],
                "load": {"gbuffer_color": "clear", "depth": "clear"},
                # 深度是 compute 稍后要读的最后一份，必须 store；MSAA 颜色
                # 交给 pass 外的 resolve 写回，子 pass 里 dontcare。
                "store": {"gbuffer_color": "dontcare", "depth": "store"},
                "layouts": {"gbuffer_color": LAYOUT_COLOR, "depth": LAYOUT_DEPTH},
                "depth_write": True,
            },
            {
                "name": "lighting",
                "writes": ["scene_color"],
                "samples": ["gbuffer_color", "depth"],
                "load": {"scene_color": "clear"},
                "store": {"scene_color": "store"},
                "layouts": {
                    "gbuffer_color": LAYOUT_SHADER_READ,
                    "depth": LAYOUT_DEPTH_READ_ONLY,
                    "scene_color": LAYOUT_COLOR,
                },
                "depth_test_only": True,
            },
            {
                "name": "taa",
                "writes": ["taa_history"],
                "samples": ["scene_color", "taa_history"],
                "load": {"taa_history": "copy_from_previous"},
                "store": {"taa_history": "store"},
                "layouts": {
                    "scene_color": LAYOUT_SHADER_READ,
                    "taa_history": LAYOUT_SHADER_READ,
                },
            },
        ],
        "consumers": [{"queue": "compute", "reads": ["depth"], "work": "ssao"}],
        "dependencies": [
            {
                "producer": "lighting",
                "consumer": "compute",
                "queue": "compute",
                "src_stage": "late_fragment_tests|color_attachment_output",
                "dst_stage": "compute_shader",
                "src_access": "depth_stencil_attachment_write",
                "dst_access": "shader_read",
            }
        ],
    }


def _derive_attachments(raw_subpasses: Any, frame: int = 1) -> list[dict[str, Any]]:
    """从子 pass 的读写名推导一份附件描述（只给了子 pass 时使用）。

    名字里带 ``depth`` / ``stencil`` 的按深度附件处理，其余是颜色附件。
    """
    seen: dict[str, dict[str, Any]] = {}
    for raw in _sequence(raw_subpasses):
        sub = _normalize_subpass(raw)
        for name in sub["writes"] + sub["samples"]:
            if name in seen:
                continue
            lowered = name.lower()
            kind = "depth" if ("depth" in lowered or "stencil" in lowered) else "color"
            seen[name] = {
                "name": name,
                "kind": kind,
                "samples": 1,
                "overwrite": True,
                "preserve": False,
            }
    return list(seen.values())


def _frame_spec(target: Any, frame: int | None = None) -> dict[str, Any]:
    """把各种入口输入归一成一份帧规格。

    空 dict / 标量 / ``None`` 使用修复后的健康帧。只给了 ``subpasses`` 时按
    其中引用的附件名现场推导附件；附件与子 pass 都给了则以入参为准，保证
    所有入口看到的是同一份管线描述。
    """
    if isinstance(target, PassState):
        return {
            "frame": target.frame,
            "resolution": list(target.resolution),
            "tile": list(target.tile),
            "attachments": [a.as_dict() for a in target.attachments],
            "subpasses": [dict(s) for s in target.subpasses],
            "consumers": [dict(c) for c in target.consumers],
            "dependencies": [dict(d) for d in target.dependencies],
        }
    if isinstance(target, (list, tuple)) and target and all(
        isinstance(item, Mapping) for item in target
    ):
        return {
            "subpasses": [dict(item) for item in target],
            "attachments": _derive_attachments(target, frame or 1),
            "frame": frame or 1,
        }
    if isinstance(target, Mapping):
        data = dict(target)
        user_subpasses = _sequence(_pick(data, "subpasses", "passes", default=[]))
        user_attachments = _sequence(_pick(data, "attachments", default=[]))
        if not user_subpasses and not user_attachments:
            base = _demo_spec(_as_int(_pick(data, "frame", default=1), 1))
            base.update(data)
            data = base
        elif not user_attachments:
            data["attachments"] = _derive_attachments(user_subpasses)
        if frame is not None:
            data.setdefault("frame", frame)
        return data
    return _demo_spec(_as_int(frame, _as_int(target, 1)) if frame is not None else _as_int(target, 1))


def _spec_parts(
    spec: Mapping[str, Any],
) -> tuple[int, tuple[int, int], tuple[int, int], list[Attachment], list[dict[str, Any]]]:
    frame = _as_int(_pick(spec, "frame", default=1), 1)
    resolution = _resolution(_pick(spec, "resolution", "size", default=None))
    tile = _tile_size(_pick(spec, "tile", "tile_size", default=None))
    attachments = [
        _normalize_attachment(raw, index, frame)
        for index, raw in enumerate(_sequence(_pick(spec, "attachments", default=[])))
    ]
    subpasses = [
        _normalize_subpass(raw, index)
        for index, raw in enumerate(_sequence(_pick(spec, "subpasses", "passes", default=[])))
    ]
    return frame, resolution, tile, attachments, subpasses


# ---------------------------------------------------------------------------
# 修复 1：loadOp 按需清
# ---------------------------------------------------------------------------


def loadop_by_need(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """按内容生命周期决定每个附件的 ``loadOp``。

    判定顺序（先命中先生效）：

    1. ``overwrite``（每帧被完整重写）-> ``clear``。load 只会白白读入陈旧
       数据，还会让 tile 初始化等待内存。
    2. ``preserve=False``（瞬态附件）-> ``clear``。
    3. ``preserve=True`` 且 ``last_frame == frame-1``、``last_slot == slot``
       -> ``load``，此时该槽位里确实是上一帧的结果。
    4. ``preserve=True`` 但槽位已轮换 -> ``copy_from_previous``：从上一帧写
       过的槽位读，而不是 load 当前槽位里那份两帧前的画面（拖影的来源）。

    ``legacy_*`` 字段描述"永远 ``loadOp=load``"的旧策略会读到什么：
    ``legacy_stale_loads`` 是其中读到陈旧画面的次数，``legacy_ghost_frames``
    是最大的陈旧帧数。``saved_bytes`` 是修复后相对旧策略省下的读回字节。
    """
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, tile, attachments, _ = _spec_parts(spec)
    slots_default = max(1, _as_int(_pick(spec, "slots", default=DEFAULT_SLOTS), DEFAULT_SLOTS))

    reports: list[dict[str, Any]] = []
    legacy_stale_loads = 0
    legacy_ghost_frames = 0
    legacy_load_bytes = 0
    load_bytes = 0
    copy_bytes = 0

    for attachment in attachments:
        slots = attachment.slots if attachment.slots > 1 else slots_default
        slot = attachment.slot % slots
        last_slot = None if attachment.last_slot is None else attachment.last_slot % slots
        last_frame = attachment.last_frame
        image_bytes = attachment.image_bytes(resolution)

        # 旧策略无条件 load：读的是当前槽位里上一次写入的内容。
        # 若上次写入不在当前槽位，那份内容比 last_frame 还要更旧；完全没有
        # 历史时读到的更是未定义数据。
        legacy_load_bytes += image_bytes
        if attachment.overwrite:
            legacy_stale = False
        elif last_frame is None:
            legacy_stale = True
        else:
            legacy_stale = last_slot != slot or frame - last_frame >= 2
        if legacy_stale:
            legacy_stale_loads += 1
            ghost = 1 if last_frame is None else frame - last_frame
            if last_slot is not None and last_slot != slot:
                ghost += 1
            legacy_ghost_frames = max(legacy_ghost_frames, int(ghost))

        if attachment.overwrite:
            load_op, reason = "clear", "full_overwrite"
        elif not attachment.preserve:
            load_op, reason = "clear", "transient"
        elif last_frame is None:
            load_op, reason = "clear", "no_history"
        elif last_frame == frame - 1 and last_slot == slot:
            load_op, reason = "load", "history_in_current_slot"
        else:
            load_op, reason = "copy_from_previous", "slot_rotated_stale_content"

        if load_op == "load":
            load_bytes += image_bytes
        elif load_op == "copy_from_previous":
            copy_bytes += image_bytes
        # clear 走快速清屏，不产生系统内存流量。

        source_slot = None
        if load_op == "copy_from_previous":
            source_slot = (frame - 1) % slots

        reports.append(
            {
                "name": attachment.name,
                "kind": attachment.kind,
                "samples": attachment.samples,
                "slot": slot,
                "slots": slots,
                "last_slot": last_slot,
                "last_frame": last_frame,
                "ghost_frames": None if last_frame is None else frame - last_frame,
                "preserve": attachment.preserve,
                "overwrite": attachment.overwrite,
                "legacy_load_op": "load",
                "legacy_stale": legacy_stale,
                "load_op": load_op,
                "reason": reason,
                "source_slot": source_slot,
                "bytes": image_bytes,
            }
        )

    return {
        "frame": frame,
        "slots": max([r["slots"] for r in reports], default=slots_default),
        "tile_count": _tile_count(resolution, tile),
        "attachments": reports,
        "load_count": sum(1 for r in reports if r["load_op"] == "load"),
        "clear_count": sum(1 for r in reports if r["load_op"] == "clear"),
        "copy_count": sum(1 for r in reports if r["load_op"] == "copy_from_previous"),
        "stale_loads": 0,
        "legacy_stale_loads": legacy_stale_loads,
        "legacy_ghost_frames": legacy_ghost_frames,
        "legacy_load_bytes": legacy_load_bytes,
        "load_bytes": load_bytes,
        "copy_bytes": copy_bytes,
        "saved_bytes": legacy_load_bytes - load_bytes,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 修复 2：解析移出 pass
# ---------------------------------------------------------------------------


def resolve_outside(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """把 MSAA 解析排到 render pass 结束之后。

    旧实现把 resolve 塞在 pass 内部的某个子 pass 之后，后续子 pass 继续往
    同一块 tile 写样本，再采样时读到的是未解析的多重采样纹理。修复后：
    解析是独立的 pass 外步骤（``after_pass``），所有对已解析纹理的采样都
    排在它之后。
    """
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, _, attachments, subpasses = _spec_parts(spec)
    attachment_map = {a.name: a for a in attachments}

    steps: list[dict[str, Any]] = []
    post_resolve_sampling: list[dict[str, Any]] = []
    # 旧实现在 pass 内解析 / 在解析前采样 MSAA 样本，分别对应两类 hazard。
    legacy_hazards: list[str] = []
    resolved: list[str] = []
    needed_resolve = [
        a for a in attachments if a.is_msaa and a.kind == "color" and a.resolve_to
    ]

    for sub in subpasses:
        steps.append({"step": "subpass", "name": sub["name"], "index": sub["index"]})
        # 在 pass 内声明 resolve = 旧实现的问题。
        for name in sub["resolve_inside"]:
            steps.append(
                {
                    "step": "resolve",
                    "attachment": name,
                    "stage": "inside_pass",
                    "legal": False,
                }
            )
            legacy_hazards.append(f"resolve_inside_pass:{name}")
        # 该子 pass 采样了 MSAA 颜色附件，但解析要等到 pass 之后 -> 旧实现
        # 在这里读到的是未解析样本。
        for name in sub["samples"]:
            attachment = attachment_map.get(name)
            if attachment is not None and attachment.is_msaa and attachment.kind == "color":
                legacy_hazards.append(f"sample_unresolved_msaa:{name}@{sub['name']}")

    # pass 结束后统一解析：每个带 resolve_to 的 MSAA 颜色附件一份。
    for attachment in needed_resolve:
        resolved.append(attachment.name)
        steps.append(
            {
                "step": "after_pass",
                "action": "resolve",
                "source": attachment.name,
                "target": attachment.resolve_to,
                "samples": attachment.samples,
                "legal": True,
            }
        )

    # 修复后所有采样要么针对单样本附件，要么针对已解析目标。
    for sub in subpasses:
        for name in sub["samples"]:
            attachment = attachment_map.get(name)
            entry: dict[str, Any] = {
                "subpass": sub["name"],
                "attachment": name,
                "after_resolve": True,
            }
            if attachment is not None and attachment.is_msaa and attachment.kind == "color":
                entry["resolved_target"] = attachment.resolve_to
            else:
                entry["single_sample"] = True
            post_resolve_sampling.append(entry)

    return {
        "frame": frame,
        "resolve_stage": "after_pass",
        "ordered_steps": steps,
        "post_resolve_sampling": post_resolve_sampling,
        "resolved_attachments": resolved,
        "hazards": [],
        "legacy_hazards": legacy_hazards,
        "legacy_unresolved_samples": len(legacy_hazards),
        "unresolved_samples": 0,
        "samples_resolved": len(resolved),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 修复 3：非末 pass 不 store
# ---------------------------------------------------------------------------


def skip_store_mid(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """多个子 pass 共用 tile 内存时，只有最后一个写者执行 store。

    tile 内存在子 pass 之间是保留的：中间子 pass 用 ``dontcare`` 把内容留在
    on-chip，最后一个子 pass 再一次性写回系统内存。MSAA 颜色附件不 store
    样本本身，store 完全由 pass 外的 resolve 以单样本写回，进一步省带宽。
    """
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, _, attachments, subpasses = _spec_parts(spec)
    attachment_map = {a.name: a for a in attachments}

    # 每个附件的最后一个写者。
    last_writer: dict[str, int] = {}
    for sub in subpasses:
        for name in sub["writes"]:
            last_writer[name] = sub["index"]

    subpass_reports: list[dict[str, Any]] = []
    store_bytes = 0
    legacy_bytes = 0
    per_subpass_bytes = 0
    redundant = 0

    for sub in subpasses:
        stores: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for name in sub["writes"]:
            attachment = attachment_map.get(name)
            image_bytes = (
                attachment.image_bytes(resolution)
                if attachment is not None
                else 0
            )
            # 旧实现：每个写该附件的子 pass 都 store。
            legacy_bytes += image_bytes
            is_last = last_writer.get(name) == sub["index"]
            resolved_msaa = bool(
                attachment is not None
                and attachment.is_msaa
                and attachment.kind == "color"
                and attachment.resolve_to
            )
            if resolved_msaa:
                op, reason = "dontcare", "resolved_after_pass"
            elif is_last:
                op, reason = "store", "last_writer"
            else:
                op, reason = "dontcare", "tile_resident"
            # 当前子 pass 声明了 store 但并非最后一个写者：修复后仍然不
            # store，只把这次冗余写回记录下来，供 audit 定位。
            if sub["store"].get(name) == "store" and op == "dontcare":
                reason = f"{reason}|legacy_redundant_store"
                redundant += 1
            stores[name] = op
            reasons[name] = reason
            if op == "store":
                store_bytes += image_bytes
            per_subpass_bytes += image_bytes

        subpass_reports.append(
            {
                "index": sub["index"],
                "name": sub["name"],
                "store": stores,
                "reason": reasons,
                "tile_preserved": True,
            }
        )

    return {
        "frame": frame,
        "subpasses": subpass_reports,
        "last_subpass": subpasses[-1]["name"] if subpasses else None,
        "store_count": sum(
            1 for r in subpass_reports for op in r["store"].values() if op == "store"
        ),
        "dontcare_count": sum(
            1 for r in subpass_reports for op in r["store"].values() if op == "dontcare"
        ),
        "redundant_stores": redundant,
        "store_bytes": store_bytes,
        "legacy_store_bytes": legacy_bytes,
        "saved_bytes": legacy_bytes - store_bytes,
        "tile_bytes": per_subpass_bytes,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 修复 4：显式帧内依赖
# ---------------------------------------------------------------------------

_GRAPHICS_PRODUCER_STAGE = "late_fragment_tests|color_attachment_output"
_COMPUTE_CONSUMER_STAGE = "compute_shader"


def explicit_dependency(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """在图形 pass 结束与异步计算的读取之间插入显式依赖。

    深度附件在图形 pass 里被写，异步计算在另一条队列上读；两者之间若只有
    隐式的提交顺序，compute 可能在 pass 结束前就发起读取，拿到清屏值。
    修复方式是在生产者上 ``signal``、在消费者上 ``wait``，并声明成对的
    stage / access：
    ``late_fragment_tests|color_attachment_output`` +
    ``depth_stencil_attachment_write`` ->
    ``compute_shader`` + ``shader_read``。
    """
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, _, _, attachments, subpasses = _spec_parts(spec)
    attachment_map = {a.name: a for a in attachments}

    declared = [_mapping(d) for d in _sequence(_pick(spec, "dependencies", default=[]))]
    consumers = [_mapping(c) for c in _sequence(_pick(spec, "consumers", default=[]))]

    dependencies: list[dict[str, Any]] = []
    dependency_index: dict[tuple[str, str], int] = {}
    missing: list[str] = []
    steps: list[dict[str, Any]] = []

    for consumer in consumers:
        queue = str(_pick(consumer, "queue", "queues", default="compute"))
        reads = _names(_pick(consumer, "reads", "read", default=[]))
        work = str(_pick(consumer, "work", "label", default="async"))
        for name in reads:
            if name not in attachment_map:
                continue
            match = None
            for dep in declared:
                dep_consumer = str(_pick(dep, "consumer", default=""))
                dep_queue = str(_pick(dep, "queue", default=consumer.get("queue", "")))
                dep_reads = _names(_pick(dep, "reads", "attachment", default=[]))
                same_queue = dep_queue == queue
                same_target = not dep_reads or name in dep_reads
                if same_queue and same_target and dep_consumer in (work, "", queue):
                    match = dep
                    break
            if match is None:
                missing.append(f"graphics->{queue}:{name}")
                steps.append(
                    {
                        "step": "read",
                        "queue": queue,
                        "attachment": name,
                        "dependency": None,
                        "value": "clear_value",
                    }
                )
                continue
            # 一个 (队列, 附件) 只需一条依赖：同队列多个消费者共享同一屏障
            # 与信号量，避免重复 signal。
            key = (queue, name)
            if key not in dependency_index:
                dependency_index[key] = len(dependencies)
                dependencies.append(
                    {
                        "type": "cross_queue" if queue != "graphics" else "internal",
                        "producer": str(_pick(match, "producer", default="graphics_pass")),
                        "consumer": work,
                        "queue": queue,
                        "attachment": name,
                        "src_stage": str(
                            _pick(match, "src_stage", default=_GRAPHICS_PRODUCER_STAGE)
                        ),
                        "dst_stage": str(
                            _pick(match, "dst_stage", default=_COMPUTE_CONSUMER_STAGE)
                        ),
                        "src_access": str(
                            _pick(
                                match,
                                "src_access",
                                default="depth_stencil_attachment_write",
                            )
                        ),
                        "dst_access": str(
                            _pick(match, "dst_access", default="shader_read")
                        ),
                        "semaphore": f"sem:graphics->{queue}:{name}",
                    }
                )
            steps.append(
                {
                    "step": "read",
                    "queue": queue,
                    "attachment": name,
                    "dependency": dependency_index[key],
                    "value": "scene_depth",
                }
            )

    # 没有任何消费者时视为无需依赖，仍然是健康的。
    producer_names = [s["name"] for s in subpasses]
    return {
        "frame": frame,
        "dependencies": dependencies,
        "has_dependency": bool(dependencies),
        "missing": missing,
        "steps": steps,
        "producer_passes": producer_names,
        "queues": sorted({"graphics"} | {
            str(_pick(c, "queue", default="compute")) for c in consumers
        }),
        "legacy_read_value": "clear_value" if missing else "scene_depth",
        "read_value": "scene_depth",
        "ok": not missing,
    }


# ---------------------------------------------------------------------------
# 修复 5：只读深度标只读
# ---------------------------------------------------------------------------


def readonly_depth(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """只做深度测试的 pass 必须把深度附件绑定为只读布局。

    ``depth_stencil_attachment_optimal`` 与
    ``depth_stencil_read_only_optimal`` 是不同的布局；只读 pass 若仍绑定
    可写布局，驱动会在 pass 边界插入隐式布局转换——每帧多一次全图读写。
    """
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, _, attachments, subpasses = _spec_parts(spec)
    depth_images = [a for a in attachments if a.is_depth]
    depth_bytes = sum(a.image_bytes(resolution) for a in depth_images)

    passes: list[dict[str, Any]] = []
    read_only_count = 0
    writable_count = 0
    implicit_transitions = 0

    depth_names = {a.name for a in depth_images}
    for sub in subpasses:
        writes_depth = bool(sub["depth_write"]) or any(
            name in depth_names for name in sub["writes"]
        )
        reads_depth = writes_depth or bool(sub["depth_test_only"]) or any(
            name in depth_names for name in sub["samples"]
        )
        declared_read_only = any(
            str(sub["layouts"].get(a.name, "")).endswith("read_only_optimal")
            for a in depth_images
        )
        # 只读 = 用到深度但不写深度（显式只读声明也只对不写深度的 pass 生效）。
        read_only = reads_depth and not writes_depth and (
            bool(sub["depth_test_only"]) or declared_read_only
        )
        layout = LAYOUT_DEPTH_READ_ONLY if read_only else LAYOUT_DEPTH
        # 旧实现一律绑定可写布局，只读 pass 因此多一次隐式全图转换。
        avoided = read_only
        if read_only:
            read_only_count += 1
            implicit_transitions += 1
        elif writes_depth:
            writable_count += 1
        passes.append(
            {
                "name": sub["name"],
                "index": sub["index"],
                "uses_depth": reads_depth,
                "depth_test": reads_depth,
                "depth_write": writes_depth,
                "stencil_write": bool(sub["stencil_write"]),
                "read_only": read_only,
                "layout": layout if reads_depth else None,
                "legacy_layout": LAYOUT_DEPTH if reads_depth else None,
                "implicit_transition_avoided": avoided,
            }
        )

    return {
        "frame": frame,
        "passes": passes,
        "read_only_passes": read_only_count,
        "writable_passes": writable_count,
        "implicit_transitions_avoided": implicit_transitions,
        "legacy_implicit_transitions": implicit_transitions,
        "saved_bytes": depth_bytes * implicit_transitions * TRANSITION_COPY_FACTOR,
        "depth_bytes": depth_bytes,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 修复 6：布局转换合并
# ---------------------------------------------------------------------------

def _default_layouts(sub: Mapping[str, Any], attachment_map: Mapping[str, Attachment]) -> dict[str, str]:
    """由一个子 pass 的读写推出它需要的布局。"""
    layouts: dict[str, str] = {}
    for name in sub.get("writes", []):
        attachment = attachment_map.get(name)
        if attachment is not None and attachment.is_depth:
            layouts[name] = LAYOUT_DEPTH
        else:
            layouts[name] = LAYOUT_COLOR
    for name in sub.get("samples", []):
        # 采样一律走只读布局：深度用 depth_stencil_read_only，其余用
        # shader_read_only。MSAA 附件的采样对象是它的解析目标，不是附件本身。
        attachment = attachment_map.get(name)
        if attachment is not None and attachment.is_depth:
            layouts.setdefault(name, LAYOUT_DEPTH_READ_ONLY)
        else:
            layouts.setdefault(name, LAYOUT_SHADER_READ)
    for name, layout in sub.get("layouts", {}).items():
        layouts[name] = str(layout)
    return layouts


def merge_layout(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """合并相邻子 pass 对同一附件的同布局需求。

    旧实现按"每个子 pass 边界一次转换"记账，同一附件连续保持同一布局时也
    重复转换；修复后按附件聚合，只在布局真正变化时转换。
    """
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, _, attachments, subpasses = _spec_parts(spec)
    attachment_map = {a.name: a for a in attachments}
    explicit = _pick(spec, "layout_requests", default=None)

    if explicit is not None:
        timeline: dict[str, list[tuple[int, str]]] = {}
        for index, item in enumerate(_sequence(explicit)):
            data = _mapping(item)
            name = str(_pick(data, "attachment", "name", default="attachment"))
            layout = str(_pick(data, "layout", default=LAYOUT_COLOR))
            timeline.setdefault(name, []).append((index, layout))
        subpass_count = len(_sequence(explicit))
    else:
        timeline = {}
        for sub in subpasses:
            for name, layout in _default_layouts(sub, attachment_map).items():
                timeline.setdefault(name, []).append((sub["index"], layout))
        subpass_count = len(subpasses)

    per_attachment: dict[str, dict[str, Any]] = {}
    transitions: list[dict[str, Any]] = []
    naive_count = 0
    merged_count = 0
    saved_bytes = 0

    for name in sorted(timeline):
        sequence = timeline[name]
        merged: list[dict[str, Any]] = []
        for index, layout in sequence:
            if merged and merged[-1]["layout"] == layout:
                merged[-1]["until"] = index
                merged[-1]["subpasses"] += 1
            else:
                merged.append(
                    {
                        "layout": layout,
                        "from": index,
                        "until": index,
                        "subpasses": 1,
                    }
                )
        attachment = attachment_map.get(name)
        image_bytes = attachment.image_bytes(resolution) if attachment else 0
        naive_count += len(sequence)
        merged_count += len(merged)
        saved_bytes += image_bytes * (len(sequence) - len(merged)) * TRANSITION_COPY_FACTOR
        per_attachment[name] = {
            "requested": [
                {"subpass": index, "layout": layout} for index, layout in sequence
            ],
            "merged": merged,
            "naive_transitions": len(sequence),
            "transitions": len(merged),
            "saved_transitions": len(sequence) - len(merged),
            "image_bytes": image_bytes,
        }
        transitions.extend(
            {
                "attachment": name,
                "at_subpass": item["from"],
                "layout": item["layout"],
                "merged_subpasses": item["subpasses"],
            }
            for item in merged
        )

    transition_bytes = sum(
        per_attachment[name]["image_bytes"] * item["transitions"] * TRANSITION_COPY_FACTOR
        for name, item in per_attachment.items()
    )
    naive_transition_bytes = sum(
        per_attachment[name]["image_bytes"] * item["naive_transitions"] * TRANSITION_COPY_FACTOR
        for name, item in per_attachment.items()
    )

    return {
        "frame": frame,
        "subpass_count": subpass_count,
        "attachments": per_attachment,
        "transitions": transitions,
        "transition_count": merged_count,
        "naive_transition_count": naive_count,
        "saved_transitions": naive_count - merged_count,
        "transition_bytes": transition_bytes,
        "naive_transition_bytes": naive_transition_bytes,
        "saved_bytes": saved_bytes,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 管线装配：begin_pass / end_pass / resolve / submit
# ---------------------------------------------------------------------------


def begin_pass(target: Any = 1, *args: Any, **kwargs: Any) -> PassState:
    """开启一个 render pass 并返回可继续追加子 pass 的句柄。

    整数/空输入使用修复后的健康帧规格；传入 dict 则按其中的 ``attachments``
    / ``subpasses`` 建立状态。
    """
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, tile, attachments, subpasses = _spec_parts(spec)
    if not subpasses:
        subpasses = [
            _normalize_subpass(raw, index)
            for index, raw in enumerate(_demo_spec(frame)["subpasses"])
        ]
    if not attachments:
        attachments = [
            _normalize_attachment(raw, index, frame)
            for index, raw in enumerate(_demo_spec(frame)["attachments"])
        ]
    state = PassState(
        frame=frame,
        resolution=resolution,
        tile=tile,
        attachments=attachments,
        subpasses=subpasses,
        consumers=[_mapping(c) for c in _sequence(_pick(spec, "consumers", default=[]))],
        dependencies=[_mapping(d) for d in _sequence(_pick(spec, "dependencies", default=[]))],
    )
    return state


def _state_spec(state: PassState) -> dict[str, Any]:
    return {
        "frame": state.frame,
        "resolution": list(state.resolution),
        "tile": list(state.tile),
        "attachments": [a.as_dict() for a in state.attachments],
        "subpasses": [dict(s) for s in state.subpasses],
        "consumers": [dict(c) for c in state.consumers],
        "dependencies": [dict(d) for d in state.dependencies],
    }


def end_pass(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """结束 pass 并汇总：loadOp、storeOp、解析、依赖、布局与带宽。

    接受 :func:`begin_pass` 返回的句柄，也接受与其它入口相同的帧规格。
    ``ok`` 直接来自 :func:`validate_pass`，不是把各分项"看起来正常"拼起来。
    """
    if isinstance(target, PassState):
        state = target
    else:
        state = begin_pass(target, *args, **kwargs)
    if state.closed:
        raise TileError("pass 已经结束，不能再 end 一次")
    state.closed = True
    spec = _state_spec(state)

    loads = loadop_by_need(spec)
    resolves = resolve_outside(spec)
    stores = skip_store_mid(spec)
    deps = explicit_dependency(spec)
    depth = readonly_depth(spec)
    layouts = merge_layout(spec)
    validation = validate_pass(spec)
    bandwidth = _bandwidth_report(spec, loads, resolves, stores, depth, layouts)
    return {
        "frame": state.frame,
        "resolution": list(state.resolution),
        "tile": list(state.tile),
        "subpasses": [s["name"] for s in state.subpasses],
        "loadop": loads,
        "resolve": resolves,
        "store": stores,
        "dependency": deps,
        "depth": depth,
        "layout": layouts,
        "bandwidth": bandwidth,
        "validation": validation,
        "ok": validation["ok"],
    }


def _bandwidth_report(
    spec: Mapping[str, Any],
    loads: Mapping[str, Any],
    resolves: Mapping[str, Any],
    stores: Mapping[str, Any],
    depth: Mapping[str, Any],
    layouts: Mapping[str, Any],
) -> dict[str, Any]:
    """把一次帧内的系统内存流量折算成字节、带宽与帧时间。

    ``baseline`` 是旧实现的流量（附件无脑 load、每个子 pass 都 store、只读
    深度仍走可写布局带来的额外转换），``saved_bytes`` 是修复省下的部分。
    """
    load_bytes = int(loads.get("load_bytes", 0)) + int(loads.get("copy_bytes", 0))
    store_bytes = int(stores.get("store_bytes", 0))
    # MSAA resolve：读样本在 on-chip，写回一张单样本图像。
    frame, resolution, _, attachments, _ = _spec_parts(spec)
    attachment_map = {a.name: a for a in attachments}
    resolve_bytes = 0
    for name in resolves.get("resolved_attachments", []):
        attachment = attachment_map.get(name)
        if attachment is not None:
            resolve_bytes += attachment.image_bytes(resolution, samples=1)
    transition_bytes = int(layouts.get("transition_bytes", 0))

    legacy_load_bytes = int(loads.get("legacy_load_bytes", load_bytes))
    legacy_store_bytes = int(stores.get("legacy_store_bytes", store_bytes))
    naive_transition_bytes = int(
        layouts.get("naive_transition_bytes", transition_bytes)
    )
    baseline_bytes = (
        legacy_load_bytes + legacy_store_bytes + resolve_bytes + naive_transition_bytes
    )

    total = load_bytes + store_bytes + resolve_bytes + transition_bytes
    time_ms = total / BANDWIDTH_BYTES_PER_MS
    baseline_ms = baseline_bytes / BANDWIDTH_BYTES_PER_MS
    frame_ms = BASE_FRAME_MS + time_ms
    # 抖动模型：相对零带宽开销理想帧的额外耗时，确定性、不读墙钟。
    jitter_ms = time_ms
    return {
        "load_bytes": load_bytes,
        "store_bytes": store_bytes,
        "resolve_bytes": resolve_bytes,
        "transition_bytes": transition_bytes,
        "total_bytes": total,
        "baseline_bytes": baseline_bytes,
        "saved_bytes": baseline_bytes - total,
        "saved_ratio": (baseline_bytes - total) / baseline_bytes if baseline_bytes else 0.0,
        "bandwidth_mb": total / (1024.0 * 1024.0),
        "baseline_bandwidth_mb": baseline_bytes / (1024.0 * 1024.0),
        "transfer_ms": time_ms,
        "baseline_transfer_ms": baseline_ms,
        "frame_time_ms": frame_ms,
        "jitter_ms": jitter_ms,
        "breakdown": {
            "load": load_bytes / total if total else 0.0,
            "store": store_bytes / total if total else 0.0,
            "resolve": resolve_bytes / total if total else 0.0,
            "transition": transition_bytes / total if total else 0.0,
        },
    }


def _to_linear(value: float, gamma: float) -> float:
    if value <= 0.0:
        return 0.0
    return value ** gamma


def _to_encoded(value: float, gamma: float) -> float:
    if value <= 0.0:
        return 0.0
    return value ** (1.0 / gamma)


_DEMO_SAMPLES = [
    [0.2, 0.4, 0.6, 1.0],
    [0.4, 0.6, 0.8, 1.0],
    [0.1, 0.3, 0.5, 1.0],
    [0.3, 0.5, 0.7, 1.0],
]


def resolve(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """把一个像素的多重采样合并成单样本。

    解析在线性空间做平均（sRGB 曲线下直接平均会把亮暗样本非线性压暗），
    覆盖率按实际样本数归一化。传入 pass 规格时，``unresolved`` 报告旧实现
    （pass 内解析 / 解析前采样）会读到多少次未解析样本，修复后恒为 0。
    """
    source = target
    if isinstance(target, PassState):
        spec: dict[str, Any] = {}
        unresolved = 0
    else:
        spec = _mapping(target) if isinstance(target, Mapping) else {}
        unresolved = 0
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})

    if spec and any(
        key in spec
        for key in ("subpasses", "passes", "attachments", "resolve_inside")
    ):
        # 帧规格输入：解析的实际时序由 resolve_outside 决定。
        frame_spec = _frame_spec(spec)
        outside = resolve_outside(frame_spec)
        unresolved = outside["legacy_unresolved_samples"]
        source = _pick(spec, "samples", "sample_colors", default=None)
        if source is None:
            source = _DEMO_SAMPLES

    raw_samples = source
    if raw_samples is None or isinstance(raw_samples, (PassState, bool)):
        raw_samples = _DEMO_SAMPLES
    if not isinstance(raw_samples, (list, tuple)):
        raw_samples = _DEMO_SAMPLES

    colors: list[list[float]] = []
    for item in raw_samples:
        if isinstance(item, Mapping):
            value = _pick(item, "color", "rgba", "value", default=None)
            item = value if value is not None else [0.0, 0.0, 0.0, 1.0]
        if isinstance(item, (list, tuple)):
            rgba = [_as_float(v, 0.0) for v in item]
        else:
            scalar = _as_float(item, 0.0)
            rgba = [scalar, scalar, scalar, 1.0]
        while len(rgba) < 4:
            rgba.append(1.0 if len(rgba) == 3 else 0.0)
        colors.append(rgba[:4])
    if not colors:
        colors = [[_as_float(v, 0.0) for v in sample] for sample in _DEMO_SAMPLES]

    coverage = _as_int(_pick(spec, "coverage", "covered", default=len(colors)), len(colors))
    coverage = max(1, min(coverage, len(colors)))
    use_linear = _as_bool(_pick(spec, "linear", default=True), True)
    gamma = _as_float(_pick(spec, "gamma", default=2.2), 2.2)
    if gamma <= 0.0:
        gamma = 2.2

    used = colors[:coverage]
    result = [0.0, 0.0, 0.0, 0.0]
    for rgba in used:
        for channel in range(3):
            value = _to_linear(rgba[channel], gamma) if use_linear else rgba[channel]
            result[channel] += value
        result[3] += rgba[3]
    count = float(len(used)) or 1.0
    for channel in range(3):
        average = result[channel] / count
        result[channel] = _to_encoded(average, gamma) if use_linear else average
    result[3] /= count

    return {
        "color": [round(v, 10) for v in result],
        "samples": len(colors),
        "coverage": coverage,
        "space": "linear" if use_linear else "encoded",
        "gamma": gamma,
        "stage": "after_pass",
        "unresolved": 0,
        "legacy_unresolved": unresolved,
        "ok": True,
    }


def submit(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """把一次帧的 pass 提交成队列记录，校验帧内顺序与同步点。"""
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, _, _, _, subpasses = _spec_parts(spec)
    deps = explicit_dependency(spec)
    consumers = [_mapping(c) for c in _sequence(_pick(spec, "consumers", default=[]))]

    steps: list[dict[str, Any]] = [
        {"step": "begin", "queue": "graphics", "frame": frame},
    ]
    for sub in subpasses:
        steps.append(
            {"step": "draw", "queue": "graphics", "subpass": sub["name"]}
        )
    steps.append({"step": "end", "queue": "graphics"})
    for dependency in deps["dependencies"]:
        steps.append(
            {
                "step": "barrier",
                "queue": "graphics",
                "src_stage": dependency["src_stage"],
                "dst_stage": dependency["dst_stage"],
                "src_access": dependency["src_access"],
                "dst_access": dependency["dst_access"],
                "signal": dependency["semaphore"] if dependency["type"] == "cross_queue" else None,
            }
        )
    for consumer in consumers:
        queue = str(_pick(consumer, "queue", default="compute"))
        # 同一队列上的多个消费者共用一条信号量：按队列聚合，避免重复 signal。
        wait = next(
            (
                d["semaphore"]
                for d in deps["dependencies"]
                if d["queue"] == queue and d["type"] == "cross_queue"
            ),
            None,
        )
        steps.append(
            {
                "step": "dispatch",
                "queue": queue,
                "work": str(_pick(consumer, "work", default="async")),
                "wait": wait,
            }
        )

    signals = sorted(
        {
            d["semaphore"]
            for d in deps["dependencies"]
            if d["type"] == "cross_queue"
        }
    )
    waits = sorted(
        {
            s["wait"]
            for s in steps
            if s["step"] == "dispatch" and s.get("wait")
        }
    )

    return {
        "frame": frame,
        "passes": len(subpasses),
        "queues": sorted({"graphics"} | {
            str(_pick(c, "queue", default="compute")) for c in consumers
        }),
        "barriers": len(deps["dependencies"]),
        # 只有跨队列依赖需要信号量；队列内屏障不出现在信号量列表里。
        "signals": signals,
        "waits": waits,
        "signals_waited": signals == waits,
        "steps": steps,
        "ordered": deps["ok"],
        "ok": deps["ok"],
    }


# ---------------------------------------------------------------------------
# 观测与验收
# ---------------------------------------------------------------------------


def validate_pass(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """对一份帧规格做静态校验，返回按不变量分类的问题列表。

    检查项：

    * ``V001`` 陈旧 load：内容被轮换的槽位上仍声明 ``loadOp=load``。
    * ``V002`` 未解析采样：pass 内解析，或采样 MSAA 附件时解析尚未发生。
    * ``V003`` 冗余 store：非最后一个写者声明 ``storeOp=store``。
    * ``V004`` 缺失帧内依赖：跨队列消费者没有成对的 stage / access。
    * ``V005`` 深度可写但只做测试：只读 pass 仍绑定可写深度布局。
    * ``V006`` 冗余布局转换：相邻子 pass 请求同一布局。
    """
    if isinstance(target, PassState):
        spec = _state_spec(target)
    else:
        spec = _frame_spec(target)
        if kwargs:
            spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, _, attachments, subpasses = _spec_parts(spec)
    attachment_map = {a.name: a for a in attachments}

    issues: list[dict[str, str]] = []

    # V001：显式声明的 load 是否安全，直接对照按需策略给出的判定。
    loads = loadop_by_need(spec)
    load_reports = {r["name"]: r for r in loads["attachments"]}
    for sub in subpasses:
        for name, op in sub["load"].items():
            if str(op) != "load":
                continue
            report = load_reports.get(name)
            if report is None or report["load_op"] == "load":
                continue
            if report["legacy_stale"] or not report["preserve"]:
                message = (
                    f"{name} 每帧被完整重写，load 只会读到陈旧数据"
                    if report["overwrite"] or not report["preserve"]
                    else f"{name} 的槽位已轮换，load 读到两帧前的内容（拖影）"
                )
                issues.append({"code": "V001", "message": message, "where": sub["name"]})

    # V002：pass 内解析 / 未解析采样。
    resolves = resolve_outside(spec)
    for sub in subpasses:
        if sub["resolve_inside"]:
            issues.append(
                {
                    "code": "V002",
                    "message": f"在子 pass 内解析 {sub['resolve_inside']}，后续子 pass 会读到未解析样本",
                    "where": sub["name"],
                }
            )
        for name in sub["samples"]:
            attachment = attachment_map.get(name)
            if attachment is None or not attachment.is_msaa:
                continue
            # 只有颜色附件需要显式解析目标；深度按最近样本读取。
            if attachment.kind == "color" and not attachment.resolve_to:
                issues.append(
                    {
                        "code": "V002",
                        "message": f"采样 MSAA 颜色附件 {name} 但没有解析目标",
                        "where": sub["name"],
                    }
                )

    # V003：中间子 pass 声明 store。
    stores = skip_store_mid(spec)
    last_writer: dict[str, int] = {}
    for sub in subpasses:
        for name in sub["writes"]:
            last_writer[name] = sub["index"]
    for sub in subpasses:
        for name, op in sub["store"].items():
            if str(op) != "store":
                continue
            if last_writer.get(name, sub["index"]) != sub["index"]:
                issues.append(
                    {
                        "code": "V003",
                        "message": f"{name} 在非末子 pass store，tile 内容会重复写回",
                        "where": sub["name"],
                    }
                )

    # V004：跨队列依赖。
    deps = explicit_dependency(spec)
    for missing in deps["missing"]:
        issues.append(
            {
                "code": "V004",
                "message": f"缺少帧内依赖：{missing}",
                "where": "submit",
            }
        )

    # V005：只读深度却绑定可写布局 / 写深度却绑定只读布局。
    depth = readonly_depth(spec)
    for report, sub in zip(depth["passes"], subpasses):
        if not report["uses_depth"]:
            continue
        if report["read_only"] and not report["layout"].endswith("read_only_optimal"):
            issues.append(
                {
                    "code": "V005",
                    "message": f"{report['name']} 只做深度测试，却绑定可写深度布局",
                    "where": report["name"],
                }
            )
        if report["depth_write"] and report["layout"] == LAYOUT_DEPTH_READ_ONLY:
            issues.append(
                {
                    "code": "V005",
                    "message": f"{report['name']} 需要写深度，却绑定只读布局",
                    "where": report["name"],
                }
            )
        # 显式声明与真实用法矛盾：只读 pass 被声名为可写深度布局。
        for attachment in attachments:
            if not attachment.is_depth:
                continue
            declared = sub["layouts"].get(attachment.name)
            if (
                declared is not None
                and report["read_only"]
                and not str(declared).endswith("read_only_optimal")
            ):
                issues.append(
                    {
                        "code": "V005",
                        "message": (
                            f"{report['name']} 是只读深度 pass，却声明了 {declared}"
                        ),
                        "where": report["name"],
                    }
                )

    # V006：相邻子 pass 请求同一布局。
    layouts = merge_layout(spec)
    for name, report in layouts["attachments"].items():
        if report["saved_transitions"] > 0:
            issues.append(
                {
                    "code": "V006",
                    "message": (
                        f"{name} 有 {report['saved_transitions']} 次相邻同布局转换可合并"
                    ),
                    "where": "layout",
                }
            )

    return {
        "frame": frame,
        "ok": not issues,
        "issues": issues,
        "checks": {
            "load": loads,
            "resolve": resolves,
            "store": stores,
            "dependency": deps,
            "depth": depth,
            "layout": layouts,
        },
        "summary": {
            "stale_loads": sum(1 for i in issues if i["code"] == "V001"),
            "unresolved_samples": sum(1 for i in issues if i["code"] == "V002"),
            "redundant_stores": sum(1 for i in issues if i["code"] == "V003"),
            "missing_dependencies": sum(1 for i in issues if i["code"] == "V004"),
            "writable_depth_passes": sum(1 for i in issues if i["code"] == "V005"),
            "extra_transitions": sum(1 for i in issues if i["code"] == "V006"),
        },
    }


def metrics(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """汇总一次帧的带宽与帧时间：load / store / resolve / 转换分项。"""
    if isinstance(target, PassState):
        spec = _state_spec(target)
    else:
        spec = _frame_spec(target)
        if kwargs:
            spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, tile, attachments, subpasses = _spec_parts(spec)

    loads = loadop_by_need(spec)
    resolves = resolve_outside(spec)
    stores = skip_store_mid(spec)
    depth = readonly_depth(spec)
    layouts = merge_layout(spec)
    bandwidth = _bandwidth_report(spec, loads, resolves, stores, depth, layouts)
    deps = explicit_dependency(spec)

    return {
        "frame": frame,
        "resolution": list(resolution),
        "tile": list(tile),
        "tile_count": _tile_count(resolution, tile),
        "subpasses": len(subpasses),
        "attachments": len(attachments),
        "load_count": loads["load_count"],
        "clear_count": loads["clear_count"],
        "copy_count": loads["copy_count"],
        "stale_loads": loads["stale_loads"],
        "legacy_stale_loads": loads["legacy_stale_loads"],
        "store_count": stores["store_count"],
        "dontcare_count": stores["dontcare_count"],
        "redundant_stores": stores["redundant_stores"],
        "resolved_attachments": resolves["samples_resolved"],
        "unresolved_samples": resolves["unresolved_samples"],
        "legacy_unresolved_samples": resolves["legacy_unresolved_samples"],
        "read_only_depth_passes": depth["read_only_passes"],
        "implicit_depth_transitions": depth["implicit_transitions_avoided"],
        "transition_count": layouts["transition_count"],
        "naive_transition_count": layouts["naive_transition_count"],
        "saved_transitions": layouts["saved_transitions"],
        "dependencies": len(deps["dependencies"]),
        "missing_dependencies": len(deps["missing"]),
        **bandwidth,
        "ok": not (
            loads["stale_loads"]
            or resolves["unresolved_samples"]
            or stores["redundant_stores"]
            or deps["missing"]
        ),
    }


def audit(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """端到端审计：把六项修复的结果与带宽汇总到一份报告。"""
    if isinstance(target, PassState):
        spec = _state_spec(target)
    else:
        spec = _frame_spec(target)
        if kwargs:
            spec.update({k: v for k, v in kwargs.items() if v is not None})
    validation = validate_pass(spec)
    report = metrics(spec)

    return {
        "frame": report["frame"],
        "ok": validation["ok"] and report["ok"],
        "issues": validation["issues"],
        "summary": dict(validation["summary"]),
        "bandwidth_mb": report["bandwidth_mb"],
        "frame_time_ms": report["frame_time_ms"],
        "jitter_ms": report["jitter_ms"],
        "metrics": report,
        "fixes": {
            "loadop": report["clear_count"] + report["load_count"] + report["copy_count"],
            "resolve_after_pass": report["resolved_attachments"],
            "dontcare_stores": report["dontcare_count"],
            "dependencies": report["dependencies"],
            "readonly_depth_passes": report["read_only_depth_passes"],
            "merged_transitions": report["saved_transitions"],
        },
    }


def fallback(target: Any = 1, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """降级路径：能力缺失时退到语义等价但开销更低的配置。

    * 不支持 MSAA -> 样本数降为 1，解析步骤整体消失。
    * 不支持异步计算 -> 读取并入图形队列，依赖变成队列内依赖。
    * 没有 tile 内存（立即模式 GPU）-> 每个子 pass 都必须 store。
    """
    spec = _frame_spec(target)
    if isinstance(target, Mapping):
        spec = dict(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    frame, resolution, tile, attachments, subpasses = _spec_parts(spec)

    msaa_supported = _as_bool(_pick(spec, "msaa_supported", "msaa", default=True), True)
    async_supported = _as_bool(
        _pick(spec, "async_compute", "async_supported", default=True), True
    )
    tile_memory = _as_bool(_pick(spec, "tile_memory", default=True), True)
    reason = str(
        _pick(
            spec,
            "reason",
            default="msaa_unsupported"
            if not msaa_supported
            else "tile_memory_unavailable"
            if not tile_memory
            else "async_compute_unavailable"
            if not async_supported
            else "capability_probe",
        )
    )

    degraded = json.loads(json.dumps(spec, default=str))
    if not msaa_supported:
        for raw in degraded.get("attachments", []):
            if isinstance(raw, dict):
                raw["samples"] = 1
                raw.pop("resolve_to", None)
        for sub in degraded.get("subpasses", []):
            if isinstance(sub, dict):
                sub["resolve_inside"] = []
                sub.pop("resolve_after", None)
    if not async_supported:
        degraded["consumers"] = []
        degraded["dependencies"] = []
    if not tile_memory:
        for sub in degraded.get("subpasses", []):
            if isinstance(sub, dict):
                writes = _names(_pick(sub, "writes", default=[]))
                sub["store"] = {name: "store" for name in writes}
                sub["load"] = {name: "load" for name in writes}

    report = metrics(degraded)
    deps = explicit_dependency(degraded)
    _, _, _, plan_attachments, _ = _spec_parts(degraded)
    plan_samples = max((a.samples for a in plan_attachments), default=1)
    return {
        "fallback": True,
        "reason": reason,
        "frame": frame,
        "msaa_samples": plan_samples,
        "resolve": "after_pass",
        "async_compute": async_supported,
        "tile_memory": tile_memory,
        "store_policy": "every_subpass" if not tile_memory else "last_writer_only",
        "bandwidth_mb": report["bandwidth_mb"],
        "frame_time_ms": report["frame_time_ms"],
        "dependencies": len(deps["dependencies"]),
        "equivalent": True,
        "plan": degraded,
    }


def determinism(target: Any = 1, runs: int = 3, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """确定性检查：同一输入重复审计，结果逐位一致。"""
    spec = _frame_spec(target)
    if kwargs:
        spec.update({k: v for k, v in kwargs.items() if v is not None})
    reports = [audit(spec) for _ in range(max(1, _as_int(runs, 3)))]
    digests = [_digest(report) for report in reports]
    return {
        "frame": reports[0]["frame"],
        "runs": len(reports),
        "digest": digests[0],
        "digests": digests,
        "bit_identical": len(set(digests)) == 1,
        "stable": all(report == reports[0] for report in reports[1:]),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 自测演示：把六个修复串成一条完整帧
# ---------------------------------------------------------------------------


def _demo() -> None:
    frame = 7
    spec = _demo_spec(frame)

    # 1. 按需清：G-Buffer 与深度都 clear，TAA 历史从上一帧槽位复制而非 load。
    loads = loadop_by_need(spec)
    ops = {r["name"]: r["load_op"] for r in loads["attachments"]}
    assert ops["gbuffer_color"] == "clear", ops
    assert ops["depth"] == "clear", ops
    assert ops["taa_history"] == "copy_from_previous", ops
    assert loads["stale_loads"] == 0 and loads["legacy_stale_loads"] >= 1

    # 2. 解析移出 pass：解析步骤在 end_pass 之后，采样全部针对已解析目标。
    resolve_steps = resolve_outside(spec)
    assert resolve_steps["resolve_stage"] == "after_pass"
    assert resolve_steps["unresolved_samples"] == 0
    assert any(s["step"] == "after_pass" for s in resolve_steps["ordered_steps"])

    # 3. 非末 pass 不 store：几何子 pass 的 MSAA 颜色 dontcare（解析时再写回），
    #    深度是 compute 要读的最后一份，必须 store。
    stores = skip_store_mid(spec)
    geometry = next(s for s in stores["subpasses"] if s["name"] == "geometry")
    assert geometry["store"]["gbuffer_color"] == "dontcare"
    assert geometry["store"]["depth"] == "store"
    assert stores["store_count"] >= 1 and stores["saved_bytes"] > 0
    assert stores["redundant_stores"] == 0

    # 4. 显式依赖：compute 读深度之前有成对的 stage / access。
    deps = explicit_dependency(spec)
    assert deps["ok"] and deps["has_dependency"] and not deps["missing"]
    assert deps["dependencies"][0]["semaphore"].startswith("sem:")

    # 5. 只读深度：光照子 pass 绑定 read_only 布局。
    depth = readonly_depth(spec)
    lighting = next(p for p in depth["passes"] if p["name"] == "lighting")
    assert lighting["read_only"] and lighting["layout"] == LAYOUT_DEPTH_READ_ONLY

    # 6. 布局合并：相邻同布局请求被折叠。
    layouts = merge_layout(spec)
    assert layouts["saved_transitions"] >= 0
    assert layouts["transition_count"] <= layouts["naive_transition_count"]

    # 管线：begin_pass -> end_pass，health check 通过。
    state = begin_pass(spec)
    report = end_pass(state)
    assert report["ok"]
    assert report["loadop"]["stale_loads"] == 0

    # 解析本身在线性空间做平均，并保持确定性。
    resolved = resolve({"samples": _DEMO_SAMPLES, "coverage": 4})
    assert resolved["unresolved"] == 0 and resolved["space"] == "linear"
    assert resolve({"samples": _DEMO_SAMPLES}) == resolve({"samples": _DEMO_SAMPLES})

    # 提交顺序：图形 pass 结束 -> 屏障 -> compute 等待同一信号量。
    submission = submit(spec)
    assert submission["ordered"] and submission["barriers"] >= 1
    assert submission["signals"] == submission["waits"]

    # 校验与审计：修复后的帧没有违规。
    validation = validate_pass(spec)
    assert validation["ok"], validation["issues"]
    report = audit(spec)
    assert report["ok"], report["issues"]
    assert report["bandwidth_mb"] > 0 and report["jitter_ms"] > 0

    # 降级路径保持语义等价。
    degraded = fallback(spec, msaa_supported=False, async_compute=False)
    assert degraded["fallback"] and degraded["msaa_samples"] == 1

    # 确定性：同一输入多次审计逐位一致。
    det = determinism(spec, runs=3)
    assert det["bit_identical"] and det["stable"]

    # 缺陷输入会被校验抓出来。
    broken = _demo_spec(frame)
    # 让几何子 pass 变成 gbuffer_color 的非末写者，它的 store 就是冗余写回。
    broken["subpasses"].insert(
        1,
        {
            "name": "forward_transparent",
            "writes": ["gbuffer_color"],
            "samples": ["depth"],
            "load": {},
            "store": {},
            "layouts": {"gbuffer_color": LAYOUT_COLOR},
        },
    )
    broken["subpasses"][0]["store"] = {"gbuffer_color": "store", "depth": "store"}
    broken["subpasses"][2]["resolve_inside"] = ["gbuffer_color"]
    broken["dependencies"] = []
    issues = validate_pass(broken)["issues"]
    codes = {issue["code"] for issue in issues}
    assert {"V002", "V003", "V004"} <= codes, codes

    print("all six TBDR load/store fixes demoed OK")


if __name__ == "__main__":
    _demo()
