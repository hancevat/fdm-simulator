import math
import sys
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QAction, QColor, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    print("PySide6 bulunamadı. Lütfen önce `pip install -r requirements.txt` komutunu çalıştırın.")
    raise exc

try:
    from pyqtgraph import Vector as PGVector
    import pyqtgraph.opengl as gl

    GL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency fallback
    gl = None
    PGVector = None
    GL_IMPORT_ERROR = exc


MODES = [
    ("intro", "Ana Sayfa"),
    ("overhang", "Overhang / Cooling"),
    ("pressure", "Pressure Advance"),
    ("input", "Input Shaping / Ringing"),
    ("retraction", "Retraction / Stringing"),
    ("flow", "Volumetric Flow"),
]

MODE_LABELS = dict(MODES)

DEFAULT_PARAMETERS = {
    "intro": {},
    "overhang": {"angle": 55, "fan": 70, "speed": 55, "support": False},
    "pressure": {"pa": 0.05, "extruder": "Direct Drive", "speed": 80},
    "input": {"acceleration": 3500, "frequency": 45, "shaper": "MZV", "speed": 100},
    "retraction": {"retraction": 1.0, "temperature": 205, "travel_speed": 160, "extruder": "Direct Drive"},
    "flow": {"layer_height": 0.20, "line_width": 0.45, "print_speed": 70, "max_flow": 12, "nozzle_diameter": "0.4"},
}

PRESET_PARAMETERS = {
    "PLA": {
        "overhang": {"angle": 50, "fan": 95, "speed": 55, "support": False},
        "pressure": {"pa": 0.05, "extruder": "Direct Drive", "speed": 80},
        "input": {"acceleration": 3500, "frequency": 45, "shaper": "MZV", "speed": 100},
        "retraction": {"retraction": 1.0, "temperature": 205, "travel_speed": 170, "extruder": "Direct Drive"},
        "flow": {"layer_height": 0.20, "line_width": 0.45, "print_speed": 70, "max_flow": 12, "nozzle_diameter": "0.4"},
    },
    "PETG": {
        "overhang": {"angle": 48, "fan": 45, "speed": 45, "support": False},
        "pressure": {"pa": 0.08, "extruder": "Direct Drive", "speed": 70},
        "input": {"acceleration": 3000, "frequency": 42, "shaper": "MZV", "speed": 90},
        "retraction": {"retraction": 1.4, "temperature": 240, "travel_speed": 150, "extruder": "Direct Drive"},
        "flow": {"layer_height": 0.20, "line_width": 0.45, "print_speed": 55, "max_flow": 10, "nozzle_diameter": "0.4"},
    },
    "ABS": {
        "overhang": {"angle": 45, "fan": 20, "speed": 45, "support": True},
        "pressure": {"pa": 0.06, "extruder": "Direct Drive", "speed": 75},
        "input": {"acceleration": 2800, "frequency": 40, "shaper": "EI", "speed": 85},
        "retraction": {"retraction": 1.2, "temperature": 245, "travel_speed": 155, "extruder": "Direct Drive"},
        "flow": {"layer_height": 0.22, "line_width": 0.48, "print_speed": 60, "max_flow": 11, "nozzle_diameter": "0.4"},
    },
}

BED_SIZE_X = 120.0
BED_SIZE_Y = 120.0
BED_THICKNESS = 1.0
PART_SCALE = 12.0
Z_VISUAL_SCALE = 6.0
LINE_WIDTH_VISUAL = 0.58
LAYER_HEIGHT_VISUAL = 0.40
NOZZLE_SCALE = 0.62


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * clamp(t, 0.0, 1.0))


def smoothstep(value):
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def color_tuple(color, alpha=None):
    qcolor = QColor(color)
    if alpha is not None:
        qcolor.setAlphaF(alpha)
    return qcolor.getRgbF()


def risk_color(risk):
    risk = clamp(risk, 0.0, 1.0)
    if risk < 0.34:
        return QColor("#3ddc84")
    if risk < 0.67:
        return QColor("#ffb020")
    return QColor("#ff4d5d")


def quality_color(quality):
    return risk_color(1.0 - clamp(quality, 0.0, 1.0))


def point_on_polyline3d(points, progress):
    points = [np.asarray(point, dtype=float) for point in points]
    if not points:
        return np.zeros(3)
    if len(points) == 1:
        return points[0]

    progress = clamp(progress, 0.0, 1.0)
    lengths = [float(np.linalg.norm(points[i + 1] - points[i])) for i in range(len(points) - 1)]
    total = sum(lengths)
    if total <= 0:
        return points[0]

    target_distance = total * progress
    walked = 0.0
    for index, length in enumerate(lengths):
        if walked + length >= target_distance:
            local = (target_distance - walked) / max(length, 0.001)
            return points[index] + (points[index + 1] - points[index]) * local
        walked += length
    return points[-1]


def partial_polyline3d(points, progress):
    points = [np.asarray(point, dtype=float) for point in points]
    if len(points) < 2:
        return np.asarray(points, dtype=float)

    progress = clamp(progress, 0.0, 1.0)
    target = point_on_polyline3d(points, progress)
    lengths = [float(np.linalg.norm(points[i + 1] - points[i])) for i in range(len(points) - 1)]
    total = sum(lengths)
    distance = total * progress
    walked = 0.0
    out = [points[0]]
    for index, length in enumerate(lengths):
        if walked + length < distance:
            out.append(points[index + 1])
            walked += length
        else:
            out.append(target)
            break
    return np.asarray(out, dtype=float)


@dataclass
class ToolpathSegment:
    start: tuple
    end: tuple
    width: float = 0.36
    height: float = 0.16
    segment_type: str = "extrusion"
    color: str = "#ff8a3d"
    risk_weight: float = 0.0
    defect_strength: float = 0.0
    meta: dict = field(default_factory=dict)

    def length(self):
        return float(np.linalg.norm(np.asarray(self.end, dtype=float) - np.asarray(self.start, dtype=float)))

    def point_at(self, progress):
        start = np.asarray(self.start, dtype=float)
        end = np.asarray(self.end, dtype=float)
        return start + (end - start) * clamp(progress, 0.0, 1.0)

    def partial(self, progress):
        progress = clamp(progress, 0.0, 1.0)
        meta = dict(self.meta)
        meta["partial_progress_start"] = float(meta.get("partial_progress_start", 0.0))
        meta["partial_progress_end"] = progress
        return ToolpathSegment(
            start=self.start,
            end=tuple(self.point_at(progress)),
            width=self.width,
            height=self.height,
            segment_type=self.segment_type,
            color=self.color,
            risk_weight=self.risk_weight,
            defect_strength=self.defect_strength,
            meta=meta,
        )


@dataclass
class ToolpathLayer:
    z_height: float
    layer_index: int
    segment_list: list = field(default_factory=list)


class PrintSimulationEngine:
    """Generates and advances simple educational FDM toolpaths."""

    def __init__(self, state):
        self.state = state
        self.visual_xy_scale = 1.0
        self.visual_z_scale = 0.42
        self.default_layer_height = 0.20
        self.mode = None
        self.signature = None
        self.layers = []
        self.flat_segments = []
        self.segment_layer_indices = []
        self.segment_durations = []
        self.total_duration = 1.0
        self.completed_segments = []
        self.active_segment = None
        self.active_segment_progress = 0.0
        self.current_layer_index = 0
        self.current_segment_index = 0
        self.nozzle_position = np.asarray((0.0, 0.0, 1.2), dtype=float)

    def visual_layer_z(self, layer_index, layer_height=None):
        layer_height = self.default_layer_height if layer_height is None else float(layer_height)
        return 0.12 + layer_index * layer_height * self.visual_z_scale

    def visual_segment_height(self, layer_height=None):
        layer_height = self.default_layer_height if layer_height is None else float(layer_height)
        return max(0.08, layer_height * self.visual_z_scale * 0.72)

    def params_signature(self):
        params = self.state.current_params()
        return (self.state.active_mode, tuple(sorted(params.items())))

    def ensure_current(self):
        signature = self.params_signature()
        if signature != self.signature:
            self.mode = self.state.active_mode
            self.signature = signature
            self.layers = self.generate_layers(self.mode, self.state.current_params())
            self.flatten_segments()
            self.update_progress(self.state.animation_time)

    def reset(self):
        self.signature = None
        self.completed_segments = []
        self.active_segment = None
        self.active_segment_progress = 0.0
        self.current_layer_index = 0
        self.current_segment_index = 0
        self.nozzle_position = np.asarray((0.0, 0.0, 1.2), dtype=float)
        self.ensure_current()

    def flatten_segments(self):
        self.flat_segments = []
        self.segment_layer_indices = []
        for layer in self.layers:
            for segment in layer.segment_list:
                self.flat_segments.append(segment)
                self.segment_layer_indices.append(layer.layer_index)
        self.segment_durations = []
        for segment in self.flat_segments:
            length = max(segment.length(), 0.05)
            base = 0.11 if segment.segment_type == "travel" else 0.22
            if segment.segment_type in {"blob", "gap"}:
                base = 0.16
            self.segment_durations.append(max(0.11, length * base))
        self.total_duration = max(sum(self.segment_durations), 0.001)

    def update_progress(self, animation_time):
        self.ensure_current()
        if not self.flat_segments:
            return
        time_in_cycle = (animation_time * 1.75) % self.total_duration
        elapsed = 0.0
        active_index = 0
        active_progress = 0.0
        for index, duration in enumerate(self.segment_durations):
            if elapsed + duration >= time_in_cycle:
                active_index = index
                active_progress = (time_in_cycle - elapsed) / max(duration, 0.001)
                break
            elapsed += duration
        else:
            active_index = len(self.flat_segments) - 1
            active_progress = 1.0

        self.current_segment_index = active_index
        self.current_layer_index = self.segment_layer_indices[active_index]
        self.active_segment_progress = clamp(active_progress, 0.0, 1.0)
        self.completed_segments = self.flat_segments[:active_index]
        self.active_segment = self.flat_segments[active_index]
        self.nozzle_position = self.active_segment.point_at(self.active_segment_progress)

    def generate_layers(self, mode, params):
        if mode == "overhang":
            return self.generate_overhang(params)
        if mode == "pressure":
            return self.generate_pressure(params)
        if mode == "input":
            return self.generate_input_shaping(params)
        if mode == "retraction":
            return self.generate_retraction(params)
        if mode == "flow":
            return self.generate_flow(params)
        return self.generate_intro(params)

    def add_rectangle_layer(self, layer, cx, cy, z, sx, sy, width=0.34, height=0.16, color="#ff8a3d", segment_type="extrusion", defect=0.0, meta=None):
        corners = [
            (cx - sx / 2, cy - sy / 2, z),
            (cx + sx / 2, cy - sy / 2, z),
            (cx + sx / 2, cy + sy / 2, z),
            (cx - sx / 2, cy + sy / 2, z),
            (cx - sx / 2, cy - sy / 2, z),
        ]
        for start, end in zip(corners[:-1], corners[1:]):
            layer.segment_list.append(ToolpathSegment(start, end, width, height, segment_type, color, defect_strength=defect, meta=dict(meta or {})))

    def add_circle_layer(self, layer, cx, cy, z, radius, width=0.24, height=0.12, color="#ff8a3d", points=24, meta=None, segment_type="extrusion", start_angle=0.0):
        path = []
        for index in range(points + 1):
            angle = start_angle + 2 * math.pi * index / points
            path.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z))
        for start, end in zip(path[:-1], path[1:]):
            layer.segment_list.append(ToolpathSegment(start, end, width, height, segment_type, color, meta=dict(meta or {})))

    def add_infill_lines(self, layer, cx, cy, z, sx, sy, count=2, width=0.28, height=0.14, color="#ff9a4d"):
        for index in range(count):
            y = cy - sy * 0.28 + index * (sy * 0.56 / max(count - 1, 1))
            layer.segment_list.append(ToolpathSegment((cx - sx * 0.36, y, z), (cx + sx * 0.36, y, z), width, height, "extrusion", color))

    def generate_intro(self, params):
        layers = []
        layer_count = 14
        for layer_index in range(layer_count):
            z = self.visual_layer_z(layer_index)
            layer = ToolpathLayer(z, layer_index)
            height = self.visual_segment_height()
            self.add_rectangle_layer(layer, 0, 0, z, 3.1, 1.75, width=0.32, height=height)
            self.add_infill_lines(layer, 0, 0, z, 3.1, 1.75, count=2, height=height * 0.9)
            layers.append(layer)
        return layers

    def generate_overhang(self, params):
        angle = float(params.get("angle", 55))
        risk = FDMModel.overhang_risk(params)
        support = bool(params.get("support", False))
        layers = []
        base_layers = 8
        overhang_layers = 10
        offset_per_layer = lerp(0.07, 0.24, (angle - 20) / 60)
        for layer_index in range(base_layers + overhang_layers):
            z = self.visual_layer_z(layer_index)
            height = self.visual_segment_height()
            layer = ToolpathLayer(z, layer_index)
            if support and layer_index < base_layers + overhang_layers - 1:
                for sx in [0.85, 1.65, 2.45]:
                    layer.segment_list.append(ToolpathSegment((sx, -0.38, z - 0.04), (sx, 0.38, z - 0.04), 0.18, height * 0.75, "support", "#7dd3fc", defect_strength=0.15))
            if layer_index < base_layers:
                self.add_rectangle_layer(layer, -1.45, 0, z, 1.65, 1.75, width=0.34, height=height)
                self.add_infill_lines(layer, -1.45, 0, z, 1.65, 1.75, count=2, height=height * 0.9)
            else:
                over_index = layer_index - base_layers
                cx = -1.00 + (over_index + 1) * offset_per_layer
                z_visual = z
                length = 1.80 + over_index * offset_per_layer
                tip_x = cx + length / 2
                self.add_rectangle_layer(
                    layer,
                    cx,
                    0,
                    z_visual,
                    length,
                    1.55,
                    width=0.30,
                    height=height,
                    color="#ff8a3d",
                    defect=risk,
                    meta={
                        "overhang_risk": risk,
                        "overhang_root_x": -0.62,
                        "overhang_tip_x": tip_x,
                        "overhang_layer_factor": (over_index + 1) / overhang_layers,
                    },
                )
            layers.append(layer)
        return layers

    def generate_pressure(self, params):
        result = FDMModel.pressure_advance_quality(params)
        low = result["low_pa_defect"]
        high = result["high_pa_defect"]
        layers = []
        for layer_index in range(12):
            z = self.visual_layer_z(layer_index)
            height = self.visual_segment_height()
            layer = ToolpathLayer(z, layer_index)
            sx = 4.3
            sy = 2.4
            corners = [
                (-sx / 2, -sy / 2, z),
                (sx / 2, -sy / 2, z),
                (sx / 2, sy / 2, z),
                (-sx / 2, sy / 2, z),
                (-sx / 2, -sy / 2, z),
            ]
            for side_index, (start, end) in enumerate(zip(corners[:-1], corners[1:])):
                meta = {
                    "profile": "pressure",
                    "pa_low": low,
                    "pa_high": high,
                    "corner_start": True,
                    "corner_end": True,
                    "side_index": side_index,
                }
                if high > 0.04:
                    layer.segment_list.append(ToolpathSegment(start, end, 0.30, height, "extrusion", "#ff8a3d", defect_strength=high, meta=meta))
                else:
                    layer.segment_list.append(ToolpathSegment(start, end, 0.30, height, "extrusion", "#ff8a3d", defect_strength=low, meta=meta))
            layers.append(layer)
        return layers

    def generate_input_shaping(self, params):
        acceleration = float(params.get("acceleration", 3500))
        frequency = float(params.get("frequency", 45))
        speed = float(params.get("speed", 100))
        shaper = params.get("shaper", "MZV")
        shaper_gain = {"Kapalı": 1.0, "MZV": 0.55, "EI": 0.35, "2HUMP_EI": 0.22}.get(shaper, 1.0)
        accel_factor = clamp((acceleration - 500) / 9500, 0, 1)
        amplitude = (0.02 + 0.34 * accel_factor) * shaper_gain
        wavelength = clamp(speed / max(frequency, 1) * 0.88, 0.24, 1.45)
        decay = lerp(0.30, 0.90, 1.0 - shaper_gain)
        layers = []
        x_points = np.linspace(-2.55, 2.55, 12)
        for layer_index in range(16):
            z = self.visual_layer_z(layer_index, 0.18)
            height = self.visual_segment_height(0.18)
            layer = ToolpathLayer(z, layer_index)
            local_x = x_points - x_points.min()
            wave = amplitude * np.sin(2 * np.pi * local_x / wavelength) * np.exp(-decay * local_x)
            front = [(float(x), float(-0.18 - abs(w) * 0.9), z) for x, w in zip(x_points, wave)]
            back = [(float(x), 0.18, z) for x in x_points[::-1]]
            path = front + back + [front[0]]
            for start, end in zip(path[:-1], path[1:]):
                layer.segment_list.append(ToolpathSegment(start, end, 0.20, height, "ringing_offset", "#ff8a3d", defect_strength=abs(start[1] + 0.18)))
            layers.append(layer)
        return layers

    def generate_retraction(self, params):
        result = FDMModel.retraction_stringing_risk(params)
        stringing = result["stringing_risk"]
        gap = result["restart_gap_risk"]
        layers = []
        centers = [(-1.8, 0.0), (1.8, 0.0)]
        for layer_index in range(16):
            z = self.visual_layer_z(layer_index)
            height = self.visual_segment_height()
            layer = ToolpathLayer(z, layer_index)
            self.add_circle_layer(
                layer,
                centers[0][0],
                centers[0][1],
                z,
                radius=0.36,
                width=0.20,
                height=height,
                points=24,
                start_angle=0.0,
                meta={"tower": "first", "layer_index": layer_index},
            )
            travel_z = z + 0.24
            travel = ToolpathSegment(
                (-1.44, 0.0, travel_z),
                (1.44, 0.0, travel_z),
                0.05,
                0.03,
                "travel",
                "#dfe7ef",
                defect_strength=stringing,
                meta={"stringing_risk": stringing, "layer_index": layer_index},
            )
            layer.segment_list.append(travel)
            self.add_circle_layer(
                layer,
                centers[1][0],
                centers[1][1],
                z,
                radius=0.36,
                width=0.20,
                height=height,
                points=24,
                start_angle=math.pi,
                meta={
                    "tower": "second",
                    "layer_index": layer_index,
                    "restart_gap_risk": gap,
                    "restart_blob_risk": clamp(stringing - 0.22, 0.0, 1.0),
                    "restart_total_length": 2 * math.pi * 0.36,
                },
            )
            layers.append(layer)
        return layers

    def generate_flow(self, params):
        result = FDMModel.volumetric_flow_risk(params)
        risk = result["risk"]
        ratio = result["ratio"]
        line_width = float(params.get("line_width", 0.45))
        layer_height = float(params.get("layer_height", 0.20))
        visual_width = lerp(0.18, 0.32, clamp((line_width - 0.35) / 0.45, 0, 1)) * lerp(1.0, 0.55, smoothstep(risk))
        layers = []
        for layer_index in range(10):
            z = self.visual_layer_z(layer_index, layer_height)
            height = self.visual_segment_height(layer_height)
            layer = ToolpathLayer(z, layer_index)
            for line_index, y in enumerate(np.linspace(-0.82, 0.82, 5)):
                start_x = -2.8
                end_x = 2.8
                if ratio <= 1.0:
                    layer.segment_list.append(ToolpathSegment((start_x, y, z), (end_x, y, z), visual_width, height, "extrusion", "#ff8a3d", defect_strength=risk, meta={"flow_ratio": ratio, "flow_risk": risk, "line_index": line_index, "layer_index": layer_index}))
                else:
                    layer.segment_list.append(ToolpathSegment((start_x, y, z), (end_x, y, z), visual_width, height * 0.78, "underextrusion", "#ff8a3d", defect_strength=risk, meta={"flow_ratio": ratio, "flow_risk": risk, "line_index": line_index, "layer_index": layer_index}))
            layers.append(layer)
        return layers


class FDMModel:
    """Temsili hesaplamalar ve yeni başlayan dostu açıklamalar."""

    @staticmethod
    def overhang_risk(params):
        angle = float(params.get("angle", 55))
        fan = float(params.get("fan", 70))
        speed = float(params.get("speed", 55))
        support = bool(params.get("support", False))
        angle_factor = clamp((angle - 45) / 35, 0, 1)
        fan_factor = 1 - fan / 100
        speed_factor = clamp((speed - 40) / 80, 0, 1)
        support_factor = 0.35 if support else 1.0
        return clamp((0.55 * angle_factor + 0.30 * fan_factor + 0.15 * speed_factor) * support_factor, 0, 1)

    @staticmethod
    def pressure_advance_quality(params):
        pa = float(params.get("pa", 0.05))
        extruder = params.get("extruder", "Direct Drive")
        speed = float(params.get("speed", 80))
        if extruder == "Direct Drive":
            ideal = 0.05
            tolerance = 0.08
        else:
            ideal = 0.35
            tolerance = 0.18
        speed_factor = clamp((speed - 60) / 120, 0, 1)
        effective_tolerance = tolerance * (1 - 0.35 * speed_factor)
        distance = abs(pa - ideal)
        quality = clamp(1 - distance / max(effective_tolerance, 0.001), 0, 1)
        low_pa_defect = clamp((ideal - pa) / max(effective_tolerance, 0.001), 0, 1)
        high_pa_defect = clamp((pa - ideal) / max(effective_tolerance, 0.001), 0, 1)
        return {
            "quality": quality,
            "low_pa_defect": low_pa_defect,
            "high_pa_defect": high_pa_defect,
            "ideal": ideal,
            "effective_tolerance": effective_tolerance,
        }

    @staticmethod
    def input_shaping_risk(params):
        acceleration = float(params.get("acceleration", 3500))
        frequency = float(params.get("frequency", 45))
        speed = float(params.get("speed", 100))
        shaper = params.get("shaper", "MZV")
        accel_factor = clamp((acceleration - 500) / 9500, 0, 1)
        speed_factor = clamp((speed - 40) / 210, 0, 1)
        frequency_factor = clamp((80 - frequency) / 60, 0.2, 1.0)
        shaper_factor = {"Kapalı": 1.0, "MZV": 0.55, "EI": 0.35, "2HUMP_EI": 0.22}.get(shaper, 1.0)
        return clamp((0.55 * accel_factor + 0.45 * speed_factor) * frequency_factor * shaper_factor, 0, 1)

    @staticmethod
    def retraction_stringing_risk(params):
        retraction = float(params.get("retraction", 1.0))
        temperature = float(params.get("temperature", 205))
        travel_speed = float(params.get("travel_speed", 160))
        extruder = params.get("extruder", "Direct Drive")
        if extruder == "Direct Drive":
            ideal = 1.0
            tolerance = 0.8
        else:
            ideal = 4.5
            tolerance = 1.5
        temp_factor = clamp((temperature - 200) / 60, 0, 1)
        travel_factor = 1 - clamp((travel_speed - 50) / 200, 0, 1)
        under_retract = clamp((ideal - retraction) / tolerance, 0, 1)
        over_retract = clamp((retraction - ideal) / tolerance, 0, 1)
        stringing_risk = clamp(0.50 * under_retract + 0.30 * temp_factor + 0.20 * travel_factor, 0, 1)
        restart_gap_risk = clamp(over_retract, 0, 1)
        combined_risk = max(stringing_risk, restart_gap_risk)
        return {
            "stringing_risk": stringing_risk,
            "restart_gap_risk": restart_gap_risk,
            "combined_risk": combined_risk,
            "ideal": ideal,
            "tolerance": tolerance,
        }

    @staticmethod
    def volumetric_flow_risk(params):
        layer_height = float(params.get("layer_height", 0.20))
        line_width = float(params.get("line_width", 0.45))
        print_speed = float(params.get("print_speed", 70))
        max_flow = float(params.get("max_flow", 12))
        flow = layer_height * line_width * print_speed
        ratio = flow / max(max_flow, 0.001)
        risk = clamp((ratio - 0.65) / 0.55, 0, 1)
        return {"flow": flow, "ratio": ratio, "risk": risk}

    @staticmethod
    def score_for_mode(mode, params):
        if mode == "pressure":
            quality = FDMModel.pressure_advance_quality(params)["quality"]
            return "Kalite Skoru", quality, int(round(quality * 100)), False
        if mode == "overhang":
            risk = FDMModel.overhang_risk(params)
        elif mode == "input":
            risk = FDMModel.input_shaping_risk(params)
        elif mode == "retraction":
            risk = FDMModel.retraction_stringing_risk(params)["combined_risk"]
        elif mode == "flow":
            risk = FDMModel.volumetric_flow_risk(params)["risk"]
        else:
            return "Hazırlık Skoru", 0.88, 88, False
        return "Risk Skoru", risk, int(round(risk * 100)), True

    @staticmethod
    def explanation_text(mode, params):
        if mode == "intro":
            return "FDM baskıda filament eritilir, nozzle'dan çıkarılır ve katman katman model oluşturulur."
        if mode == "overhang":
            risk = FDMModel.overhang_risk(params)
            if risk < 0.34:
                return "Overhang güvenli görünüyor; fan ve hız ayarları bu açı için yeterli."
            if risk < 0.67:
                return "Sarkma başlayabilir; fanı artırmak veya baskı hızını düşürmek yardımcı olur."
            return "Sarkma riski yüksek; support, daha güçlü fan veya daha düşük hız önerilir."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "PA değeri idealin altında; köşelerde blob ve şişme görülebilir."
            if result["high_pa_defect"] > 0.33:
                return "PA fazla; köşe yakınlarında incelme veya küçük boşluklar oluşabilir."
            return "PA ideal aralığa yakın; köşe basıncı daha dengeli görünür."
        if mode == "input":
            if params.get("shaper", "MZV") == "Kapalı":
                return "Input shaping kapalıyken hız ve ivme arttıkça ringing izleri belirginleşir."
            return "Input shaping titreşim izlerini azaltır; agresif shaper ayarları smoothing etkisi yaratabilir."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            if result["restart_gap_risk"] > 0.35:
                return "Retraction fazla; ikinci kule başlangıcında restart gap görülebilir."
            if result["stringing_risk"] > 0.35:
                return "Retraction düşük veya sıcaklık yüksek; travel sırasında stringing artabilir."
            return "Retraction değeri dengeli; travel sırasında sızıntı kontrollü görünüyor."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] <= 0.75:
                return "Hotend bu debiyi rahat karşılıyor; ekstrüzyon çizgisi dolu görünür."
            if result["ratio"] <= 1.0:
                return "Flow hotend limitine yaklaşıyor; küçük bir güvenlik payı bırakmak iyi olur."
            return "Hotend kapasitesi aşılıyor; çizgide incelme ve düzenli kopmalar görülebilir."
        return ""

    @staticmethod
    def recommendation_text(mode, params):
        if mode == "intro":
            return "Soldan bir mod seç, parametreleri değiştir ve 3D sahnedeki temsili sonucu izle."
        if mode == "overhang":
            return "Sarkma varsa fanı artır, hızı azalt veya support kullan. ABS gibi malzemelerde fan notunu ayrıca değerlendir."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "Köşe şişiyorsa PA değerini küçük adımlarla artır."
            if result["high_pa_defect"] > 0.33:
                return "Köşe öncesi boşluk varsa PA değerini azalt."
            return "Bu aralık iyi görünüyor; farklı hızlarda küçük test baskısıyla doğrula."
        if mode == "input":
            return "Ringing belirginse acceleration azalt veya MZV/EI/2HUMP_EI shaper tiplerini karşılaştır."
        if mode == "retraction":
            return "Stringing varsa retraction, sıcaklık ve travel hızını birlikte ayarla; gap varsa retraction mesafesini azalt."
        if mode == "flow":
            return "Debi yüksekse hız, layer height veya line width azaltılabilir."
        return ""

    @staticmethod
    def formula_text(mode, params):
        if mode == "intro":
            return "Temsil notu: 3D sahne eğitim amaçlı ölçeklendirilmiştir; gerçek fizik motoru kullanılmaz."
        if mode == "overhang":
            return "Risk = açı, fan, hız ve support etkilerinin ağırlıklı toplamı."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            return f"İdeal PA: {result['ideal']:.2f}, tolerans: ±{result['effective_tolerance']:.2f}."
        if mode == "input":
            return "Dalga = amplitude × sin(2πx / wavelength) × exp(-decay × x)."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            return f"İdeal retraction: {result['ideal']:.1f} mm; üstünde restart gap, altında stringing artar."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            return f"Flow = Layer Height × Line Width × Speed = {result['flow']:.2f} mm³/s."
        return ""

    @staticmethod
    def calculated_value_text(mode, params):
        if mode == "overhang":
            return f"Sarkma riski: {FDMModel.overhang_risk(params) * 100:.0f}%"
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            return f"PA kalite: {result['quality'] * 100:.0f}% | ideal {result['ideal']:.2f}"
        if mode == "input":
            return f"Ringing riski: {FDMModel.input_shaping_risk(params) * 100:.0f}%"
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            return f"Stringing {result['stringing_risk'] * 100:.0f}% | Gap {result['restart_gap_risk'] * 100:.0f}%"
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            return f"{result['flow']:.2f} / {float(params.get('max_flow', 12)):.1f} mm³/s | Ratio {result['ratio']:.2f}"
        return "3D öğretici görünüm hazır"

    @staticmethod
    def progress_items(mode, params):
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            return [("Köşe kalite skoru", result["quality"], False)]
        if mode == "overhang":
            return [("Sarkma riski", FDMModel.overhang_risk(params), True)]
        if mode == "input":
            return [("Ringing riski", FDMModel.input_shaping_risk(params), True)]
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            return [("Genel risk", result["combined_risk"], True)]
        if mode == "flow":
            return [("Hotend limit riski", FDMModel.volumetric_flow_risk(params)["risk"], True)]
        return [("Rehber tamamlık", 0.88, False)]

    @staticmethod
    def report_copy_text(mode, params):
        score_label, _, score, _ = FDMModel.score_for_mode(mode, params)
        lines = [
            "FDM Parametreleri Görselleştiricisi",
            f"Mod: {MODE_LABELS.get(mode, mode)}",
            f"{score_label}: {score}/100",
            f"Hesaplanan değer: {FDMModel.calculated_value_text(mode, params)}",
            "",
            "Parametreler:",
        ]
        if params:
            for key, value in params.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- Ana sayfa: parametre yok")
        lines.extend(
            [
                "",
                f"Açıklama: {FDMModel.explanation_text(mode, params)}",
                f"Tavsiye: {FDMModel.recommendation_text(mode, params)}",
                f"Formül/Metod: {FDMModel.formula_text(mode, params)}",
            ]
        )
        return "\n".join(lines)


class SimulationState:
    def __init__(self):
        self.active_mode = "intro"
        self.running = True
        self.animation_time = 0.0
        self.animation_speed = 1.0
        self.selected_preset = "PLA"
        self.parameters = deepcopy(DEFAULT_PARAMETERS)
        self.apply_preset("PLA")

    def reset(self):
        self.animation_time = 0.0

    def set_mode(self, mode):
        if mode in MODE_LABELS:
            self.active_mode = mode
            self.animation_time = 0.0

    def update_parameter(self, key, value):
        self.parameters.setdefault(self.active_mode, {})
        self.parameters[self.active_mode][key] = value

    def current_params(self):
        return self.parameters.get(self.active_mode, {})

    def apply_preset(self, preset_name):
        self.selected_preset = preset_name
        if preset_name == "Custom":
            return
        preset = PRESET_PARAMETERS.get(preset_name, {})
        for mode, values in preset.items():
            self.parameters.setdefault(mode, {})
            self.parameters[mode].update(values)


class GLSceneWidget(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.engine = PrintSimulationEngine(state)
        self.items = []
        self.camera_mode = None
        self.setObjectName("GLSceneShell")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if gl is None:
            self.view = None
            error = QLabel(
                "3D sahne için PyOpenGL gerekiyor.\n"
                "Lütfen `pip install -r requirements.txt` komutunu çalıştırın.\n\n"
                f"Ayrıntı: {GL_IMPORT_ERROR}"
            )
            error.setAlignment(Qt.AlignCenter)
            error.setWordWrap(True)
            error.setObjectName("ErrorBox")
            layout.addWidget(error)
            return

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#0b1118")
        layout.addWidget(self.view)
        self.update_mode_scene(reset_camera=True)

    def setup_camera(self, mode=None):
        if self.view is None:
            return
        mode = mode or self.state.active_mode
        presets = {
            "intro": ((0, 0, 4.0), 155.0, 28, -42),
            "overhang": ((6, 0, 4.5), 165.0, 20, -66),
            "pressure": ((0, 0, 3.2), 150.0, 50, -40),
            "input": ((0, -1.8, 4.8), 138.0, 22, -74),
            "retraction": ((0, 0, 4.8), 155.0, 28, -42),
            "flow": ((0, 0, 2.8), 138.0, 38, -46),
        }
        center, distance, elevation, azimuth = presets.get(mode, presets["intro"])
        self.view.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)
        if PGVector is not None:
            self.view.opts["center"] = PGVector(*center)
        self.camera_mode = mode

    def clear_scene(self):
        if self.view is None:
            return
        self.view.clear()
        self.items = []

    def add_item(self, item):
        if self.view is None or item is None:
            return item
        self.view.addItem(item)
        self.items.append(item)
        return item

    def add_grid(self):
        if self.view is None:
            return
        grid = gl.GLGridItem(color=(92, 108, 120, 46))
        grid.setSize(x=BED_SIZE_X, y=BED_SIZE_Y, z=1)
        grid.setSpacing(x=10, y=10, z=1)
        self.add_item(grid)

    def add_print_bed(self):
        self.add_box(center=(0, 0, -BED_THICKNESS / 2), size=(BED_SIZE_X, BED_SIZE_Y, BED_THICKNESS), color="#202b35", alpha=1.0, edge="#3a4a59")
        x = BED_SIZE_X / 2
        y = BED_SIZE_Y / 2
        z = 0.06
        self.add_line([(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z), (-x, -y, z)], "#607384", 1.0, 0.85)

    def scene_point(self, point):
        point = np.asarray(point, dtype=float)
        return np.asarray((point[0] * PART_SCALE, point[1] * PART_SCALE, point[2] * Z_VISUAL_SCALE), dtype=float)

    def scene_width(self, width):
        return max(0.65, float(width) * PART_SCALE * LINE_WIDTH_VISUAL)

    def scene_height(self, height):
        return max(0.22, float(height) * Z_VISUAL_SCALE * LAYER_HEIGHT_VISUAL)

    def mesh_item(self, vertexes, faces, color="#ff8a3d", alpha=1.0, edge="#101820", gl_options=None, draw_edges=True, face_colors=None):
        if gl is None:
            return None
        if face_colors is None:
            face_colors = np.tile(np.asarray(color_tuple(color, alpha), dtype=np.float32), (len(faces), 1))
        else:
            face_colors = np.asarray(face_colors, dtype=np.float32)
        return gl.GLMeshItem(
            vertexes=np.asarray(vertexes, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.uint32),
            faceColors=face_colors,
            smooth=False,
            drawEdges=draw_edges,
            edgeColor=color_tuple(edge, min(0.72, 0.12 + alpha * 0.55)),
            computeNormals=False,
            glOptions=gl_options or ("translucent" if alpha < 1 else "opaque"),
        )

    def shaded_color(self, color, alpha=1.0, shade=100):
        qcolor = QColor(color)
        if shade > 100:
            qcolor = qcolor.lighter(shade)
        elif shade < 100:
            qcolor = qcolor.darker(int(10000 / max(shade, 1)))
        qcolor.setAlphaF(alpha)
        return qcolor.getRgbF()

    def is_bead_segment(self, segment):
        return segment.segment_type in {"extrusion", "support", "underextrusion", "ringing_offset"}

    def segments_connect(self, previous, current):
        if not (self.is_bead_segment(previous) and self.is_bead_segment(current)):
            return False
        if previous.segment_type != current.segment_type or previous.color != current.color:
            return False
        previous_end = np.asarray(previous.end, dtype=float)
        current_start = np.asarray(current.start, dtype=float)
        if abs(previous_end[2] - current_start[2]) > 0.08:
            return False
        return float(np.linalg.norm(previous_end - current_start)) < 0.10

    def collect_bead_runs(self, segments):
        runs = []
        current = []
        for segment in segments:
            if not self.is_bead_segment(segment):
                if current:
                    runs.append(current)
                    current = []
                runs.append(segment)
                continue
            if current and not self.segments_connect(current[-1], segment):
                runs.append(current)
                current = []
            current.append(segment)
        if current:
            runs.append(current)
        return runs

    def bead_run_points(self, run):
        points = [np.asarray(run[0].start, dtype=float)]
        for segment in run:
            start = np.asarray(segment.start, dtype=float)
            end = np.asarray(segment.end, dtype=float)
            if float(np.linalg.norm(points[-1] - start)) > 0.04:
                points.append(start)
            points.append(end)
        return points

    def densify_path_points(self, points, max_step=0.32):
        if len(points) < 2:
            return points
        dense = [np.asarray(points[0], dtype=float)]
        for start, end in zip(points[:-1], points[1:]):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            length = float(np.linalg.norm(end - start))
            steps = max(1, int(math.ceil(length / max(max_step, 0.04))))
            for step in range(1, steps + 1):
                dense.append(start + (end - start) * (step / steps))
        return dense

    def path_distances(self, points):
        distances = [0.0]
        for start, end in zip(points[:-1], points[1:]):
            distances.append(distances[-1] + float(np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float))))
        return distances

    def deterministic_noise(self, index, seed=0):
        value = math.sin((index + 1) * 12.9898 + (seed + 1) * 78.233) * 43758.5453
        return value - math.floor(value)

    def generate_bead_samples_along_path(self, run, max_step=0.14):
        points = []
        samples = []
        distance_offset = 0.0
        for segment_index, segment in enumerate(run):
            start = np.asarray(segment.start, dtype=float)
            end = np.asarray(segment.end, dtype=float)
            length = max(float(np.linalg.norm(end - start)), 0.001)
            steps = max(1, min(80, int(math.ceil(length / max(max_step, 0.04)))))
            progress_start = float(segment.meta.get("partial_progress_start", 0.0))
            progress_end = float(segment.meta.get("partial_progress_end", 1.0))
            first_step = 0 if not points else 1
            for step in range(first_step, steps + 1):
                t = step / steps
                point = start + (end - start) * t
                original_t = clamp(progress_start + (progress_end - progress_start) * t, 0.0, 1.0)
                points.append(point)
                samples.append(
                    {
                        "segment": segment,
                        "segment_index": segment_index,
                        "sample_index": len(samples),
                        "local_progress": original_t,
                        "visible_progress": t,
                        "distance": distance_offset + length * t,
                        "point": point,
                    }
                )
            distance_offset += length
        return points, samples, max(distance_offset, 0.001)

    def sample_gap_range(self, samples, index, total, scale=0.78):
        center = samples[index]["distance"]
        if len(samples) <= 1:
            half = total * 0.010
        else:
            previous_distance = samples[max(0, index - 1)]["distance"]
            next_distance = samples[min(len(samples) - 1, index + 1)]["distance"]
            half = max((next_distance - previous_distance) * 0.5 * scale, total * 0.003)
        return (clamp((center - half) / total, 0.0, 1.0), clamp((center + half) / total, 0.0, 1.0))

    def corner_profile_influence(self, sample, zone=0.18):
        segment = sample["segment"]
        local = sample["local_progress"]
        influence = 0.0
        if segment.meta.get("corner_start", False):
            influence = max(influence, smoothstep(1.0 - local / max(zone, 0.01)))
        if segment.meta.get("corner_end", False):
            influence = max(influence, smoothstep(1.0 - (1.0 - local) / max(zone, 0.01)))
        return influence

    def apply_corner_blob_profile(self, modifiers, samples):
        for index, sample in enumerate(samples):
            meta = sample["segment"].meta
            influence = self.corner_profile_influence(sample)
            if influence <= 0.0:
                continue
            low = float(meta.get("pa_low", 0.0))
            high = float(meta.get("pa_high", 0.0))
            if low > 0.01:
                modifiers[index] *= 1.0 + low * influence * 1.20
            if high > 0.01:
                modifiers[index] *= max(0.30, 1.0 - high * influence * 0.70)
        return modifiers

    def apply_underextrusion_profile(self, modifiers, samples):
        for index, sample in enumerate(samples):
            segment = sample["segment"]
            meta = segment.meta
            ratio = float(meta.get("flow_ratio", 0.0))
            if ratio <= 0.75 and segment.segment_type != "underextrusion":
                continue
            risk = clamp(float(meta.get("flow_risk", segment.defect_strength)), 0.0, 1.0)
            severity = clamp((ratio - 0.75) / 0.75, 0.0, 1.0)
            seed = int(meta.get("line_index", 0)) * 37 + int(meta.get("layer_index", 0)) * 19
            noise = self.deterministic_noise(index, seed)
            slow_pulse = 0.5 + 0.5 * math.sin(2 * math.pi * (sample["visible_progress"] * 2.7 + noise * 0.33))
            thinning = severity * (0.10 + 0.22 * noise + 0.12 * slow_pulse) + risk * 0.08
            modifiers[index] *= max(0.24, 1.0 - thinning)
        return modifiers

    def apply_restart_profile(self, modifiers, samples, total):
        if total <= 0:
            return modifiers
        for index, sample in enumerate(samples):
            meta = sample["segment"].meta
            gap_risk = clamp(float(meta.get("restart_gap_risk", 0.0)), 0.0, 1.0)
            blob_risk = clamp(float(meta.get("restart_blob_risk", 0.0)), 0.0, 1.0)
            if gap_risk <= 0.01 and blob_risk <= 0.01:
                continue
            profile_total = max(float(meta.get("restart_total_length", total)), 0.001)
            seam_distance = min(sample["distance"], abs(profile_total - sample["distance"]))
            influence = smoothstep(1.0 - seam_distance / max(profile_total * 0.13, 0.001))
            if gap_risk > 0.01:
                modifiers[index] *= max(0.22, 1.0 - gap_risk * influence * 0.78)
            if blob_risk > 0.01:
                modifiers[index] *= 1.0 + blob_risk * influence * 0.55
        return modifiers

    def apply_width_profile(self, samples, total):
        modifiers = np.ones(len(samples), dtype=float)
        modifiers = self.apply_corner_blob_profile(modifiers, samples)
        modifiers = self.apply_underextrusion_profile(modifiers, samples)
        modifiers = self.apply_restart_profile(modifiers, samples, total)
        return np.clip(modifiers, 0.18, 1.95)

    def apply_gap_profile(self, samples, total):
        gaps = []
        if total <= 0 or not samples:
            return gaps

        pressure_gap_added = set()
        for index, sample in enumerate(samples):
            segment = sample["segment"]
            meta = segment.meta
            high = clamp(float(meta.get("pa_high", 0.0)), 0.0, 1.0)
            influence = self.corner_profile_influence(sample, zone=0.12)
            near_start = sample["local_progress"] < 0.055
            near_end = sample["local_progress"] > 0.945
            corner_key = (sample["segment_index"], "start" if near_start else "end")
            if high > 0.68 and influence > 0.72 and (near_start or near_end) and corner_key not in pressure_gap_added:
                gaps.append(self.sample_gap_range(samples, index, total, 0.58 + high * 0.20))
                pressure_gap_added.add(corner_key)

            ratio = float(meta.get("flow_ratio", 0.0))
            if ratio > 1.0:
                risk = clamp(float(meta.get("flow_risk", segment.defect_strength)), 0.0, 1.0)
                seed = int(meta.get("line_index", 0)) * 41 + int(meta.get("layer_index", 0)) * 23
                noise = self.deterministic_noise(index, seed)
                gap_probability = clamp((ratio - 1.0) * 0.075 + risk * 0.055, 0.0, 0.16)
                away_from_ends = 0.035 < sample["visible_progress"] < 0.965
                if away_from_ends and noise < gap_probability:
                    gaps.append(self.sample_gap_range(samples, index, total, 0.48 + risk * 0.35))

        restart_segments = [sample for sample in samples if float(sample["segment"].meta.get("restart_gap_risk", 0.0)) > 0.14]
        if restart_segments:
            gap_risk = max(float(sample["segment"].meta.get("restart_gap_risk", 0.0)) for sample in restart_segments)
            width = clamp(0.010 + gap_risk * 0.040, 0.012, 0.060)
            gaps.append((0.0, width))
            gaps.append((1.0 - width * 0.55, 1.0))

        return gaps

    def apply_z_offset_profile(self, samples):
        offsets = np.zeros(len(samples), dtype=float)
        for index, sample in enumerate(samples):
            meta = sample["segment"].meta
            risk = clamp(float(meta.get("overhang_risk", 0.0)), 0.0, 1.0)
            if risk <= 0.01:
                continue
            root_x = float(meta.get("overhang_root_x", sample["point"][0]))
            tip_x = float(meta.get("overhang_tip_x", root_x + 0.001))
            layer_factor = clamp(float(meta.get("overhang_layer_factor", 1.0)), 0.0, 1.0)
            free_end_factor = clamp((sample["point"][0] - root_x) / max(tip_x - root_x, 0.001), 0.0, 1.0)
            max_sag = 0.045 + 0.075 * layer_factor
            offsets[index] -= risk * (free_end_factor ** 1.55) * max_sag
        return offsets

    def corner_strength(self, previous, current, following):
        previous = np.asarray(previous, dtype=float)
        current = np.asarray(current, dtype=float)
        following = np.asarray(following, dtype=float)
        incoming = current[:2] - previous[:2]
        outgoing = following[:2] - current[:2]
        in_len = float(np.linalg.norm(incoming))
        out_len = float(np.linalg.norm(outgoing))
        if in_len < 0.001 or out_len < 0.001:
            return 0.0
        incoming /= in_len
        outgoing /= out_len
        dot = clamp(float(np.dot(incoming, outgoing)), -1.0, 1.0)
        return clamp((1.0 - dot) * 0.5, 0.0, 1.0)

    def bead_modifiers(self, points, run):
        count = len(points)
        modifiers = np.ones(count, dtype=float)
        gaps = []
        if count < 2:
            return modifiers, gaps

        distances = self.path_distances(points)
        total = max(distances[-1], 0.001)
        closed = float(np.linalg.norm(np.asarray(points[0]) - np.asarray(points[-1]))) < 0.001
        pa_low = max(float(segment.meta.get("pa_low", 0.0)) for segment in run)
        pa_high = max(float(segment.meta.get("pa_high", 0.0)) for segment in run)
        flow_risk = max(float(segment.meta.get("flow_risk", segment.defect_strength)) for segment in run)
        flow_ratio = max(float(segment.meta.get("flow_ratio", 0.0)) for segment in run)
        underextrusion = any(segment.segment_type == "underextrusion" for segment in run)

        indices = range(count - 1) if closed else range(1, count - 1)
        for index in indices:
            if closed:
                previous = points[index - 1 if index > 0 else count - 2]
                following = points[(index + 1) % (count - 1)]
            else:
                previous = points[index - 1]
                following = points[index + 1]
            current = points[index]
            turn = self.corner_strength(previous, current, following)
            if turn <= 0.04:
                continue
            if pa_low > 0.04:
                modifiers[index] *= 1.0 + turn * pa_low * 0.85
            if pa_high > 0.04:
                modifiers[index] *= max(0.38, 1.0 - turn * pa_high * 0.72)
            if pa_high > 0.72 and turn > 0.30:
                center = distances[index] / total
                half_width = 0.010 + pa_high * 0.010
                gaps.append((clamp(center - half_width, 0.0, 1.0), clamp(center + half_width, 0.0, 1.0)))

        if closed:
            modifiers[-1] = modifiers[0]

        if underextrusion or flow_ratio > 1.0:
            risk = clamp(max(flow_risk, flow_ratio - 1.0), 0.0, 1.0)
            base = max(0.48, 1.0 - risk * 0.34)
            for index, distance in enumerate(distances):
                t = distance / total
                wave = 0.5 + 0.5 * math.sin(2 * math.pi * (t * 7.0 + 0.15))
                modifiers[index] *= base * lerp(0.70, 1.0, wave)
            if flow_ratio > 1.08 or risk > 0.35:
                gap_step = lerp(0.18, 0.12, risk)
                gap_half = lerp(0.004, 0.012, risk)
                center = gap_step * 0.7
                while center < 0.98:
                    gaps.append((clamp(center - gap_half, 0.0, 1.0), clamp(center + gap_half, 0.0, 1.0)))
                    center += gap_step

        return np.clip(modifiers, 0.34, 1.75), gaps

    def fillet_path(self, points, modifiers, fillet_distance):
        if len(points) < 3:
            return points, modifiers

        closed = float(np.linalg.norm(np.asarray(points[0]) - np.asarray(points[-1]))) < 0.04
        source_points = points[:-1] if closed else points
        source_modifiers = modifiers[:-1] if closed else modifiers
        out_points = []
        out_modifiers = []
        count = len(source_points)

        def append_point(point, modifier):
            if out_points and float(np.linalg.norm(np.asarray(point) - np.asarray(out_points[-1]))) < 0.02:
                out_points[-1] = point
                out_modifiers[-1] = modifier
            else:
                out_points.append(point)
                out_modifiers.append(modifier)

        for index, current in enumerate(source_points):
            if not closed and (index == 0 or index == count - 1):
                append_point(current, source_modifiers[index])
                continue

            previous = source_points[index - 1]
            following = source_points[(index + 1) % count]
            turn = self.corner_strength(previous, current, following)
            if turn < 0.12:
                append_point(current, source_modifiers[index])
                continue

            incoming = np.asarray(current) - np.asarray(previous)
            outgoing = np.asarray(following) - np.asarray(current)
            in_len = float(np.linalg.norm(incoming))
            out_len = float(np.linalg.norm(outgoing))
            if in_len < 0.05 or out_len < 0.05:
                append_point(current, source_modifiers[index])
                continue

            incoming /= in_len
            outgoing /= out_len
            distance = min(fillet_distance, in_len * 0.42, out_len * 0.42)
            before = np.asarray(current) - incoming * distance
            after = np.asarray(current) + outgoing * distance
            middle = np.asarray(current) + (outgoing - incoming) * distance * 0.42
            modifier = source_modifiers[index]
            append_point(before, lerp(source_modifiers[index - 1], modifier, 0.65))
            append_point(middle, modifier)
            append_point(after, lerp(modifier, source_modifiers[(index + 1) % count], 0.35))

        if closed and out_points:
            out_points.append(np.asarray(out_points[0], dtype=float))
            out_modifiers.append(out_modifiers[0])

        return out_points, np.asarray(out_modifiers, dtype=float)

    def in_gap_range(self, value, gap_ranges):
        return any(start <= value <= end for start, end in gap_ranges)

    def create_extrusion_bead_mesh(self, path_points, width, height, color, width_modifiers=None, gap_ranges=None, z_offsets=None, alpha=1.0):
        if gl is None:
            return None
        points = [np.asarray(point, dtype=float) for point in path_points]
        if z_offsets is not None:
            offsets = np.zeros(len(points), dtype=float)
            source_offsets = np.asarray(z_offsets, dtype=float)
            offsets[: min(len(offsets), len(source_offsets))] = source_offsets[: min(len(offsets), len(source_offsets))]
            points = [point + np.asarray((0.0, 0.0, offsets[index]), dtype=float) for index, point in enumerate(points)]
        if len(points) < 2:
            if points:
                return self.add_sphere_marker(points[0], 6.0, color, alpha)
            return None

        if width_modifiers is None:
            modifier_source = np.ones(len(points), dtype=float)
        else:
            modifier_source = np.ones(len(points), dtype=float)
            source_modifiers = np.asarray(width_modifiers, dtype=float)
            modifier_source[: min(len(modifier_source), len(source_modifiers))] = source_modifiers[: min(len(modifier_source), len(source_modifiers))]

        filtered_points = [points[0]]
        filtered_modifiers = [modifier_source[0]]
        for index, point in enumerate(points[1:], start=1):
            if float(np.linalg.norm(point - filtered_points[-1])) > 0.025:
                filtered_points.append(point)
                filtered_modifiers.append(modifier_source[index])
        points = filtered_points
        modifiers = np.asarray(filtered_modifiers, dtype=float)
        if len(points) < 2:
            return None

        gap_ranges = gap_ranges or []
        distances = self.path_distances(points)
        total = max(distances[-1], 0.001)
        closed = float(np.linalg.norm(points[0] - points[-1])) < 0.04
        vertices = []
        for index, point in enumerate(points):
            if index == 0:
                tangent = points[1] - point
            elif index == len(points) - 1:
                tangent = point - points[index - 1]
            else:
                tangent = points[index + 1] - points[index - 1]
            tangent_xy = tangent[:2]
            tangent_len = float(np.linalg.norm(tangent_xy))
            if tangent_len < 0.001:
                tangent_xy = np.asarray((1.0, 0.0), dtype=float)
                tangent_len = 1.0
            tangent_xy /= tangent_len
            perp = np.asarray((-tangent_xy[1], tangent_xy[0], 0.0), dtype=float)
            local_width = width * clamp(float(modifiers[index]), 0.30, 1.85)
            half = local_width * 0.5
            z = np.asarray((0.0, 0.0, 1.0), dtype=float)
            vertices.extend(
                [
                    point - perp * half - z * height * 0.40,
                    point + perp * half - z * height * 0.40,
                    point + perp * half * 0.96 + z * height * 0.02,
                    point + perp * half * 0.34 + z * height * 0.46,
                    point - perp * half * 0.34 + z * height * 0.46,
                    point - perp * half * 0.96 + z * height * 0.02,
                ]
            )

        faces = []
        face_colors = []
        side_dark = self.shaded_color(color, alpha, 78)
        side_mid = self.shaded_color(color, alpha, 94)
        top_light = self.shaded_color(color, alpha, 116)
        palette = [side_dark, side_mid, side_mid, top_light, side_mid, side_mid]

        def add_tri(face, rgba):
            faces.append(face)
            face_colors.append(rgba)

        def add_quad(a, b, c, d, rgba):
            add_tri((a, b, d), rgba)
            add_tri((b, c, d), rgba)

        def add_cap(section, reverse=False):
            base = section * 6
            rgba = side_dark
            for offset in range(1, 5):
                if reverse:
                    add_tri((base, base + offset + 1, base + offset), rgba)
                else:
                    add_tri((base, base + offset, base + offset + 1), rgba)

        pair_count = len(points) - 1
        visible = []
        for index in range(pair_count):
            midpoint = ((distances[index] + distances[index + 1]) * 0.5) / total
            visible.append(not self.in_gap_range(midpoint, gap_ranges))

        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            previous_visible = visible[index - 1] if index > 0 else (visible[-1] if closed else False)
            next_visible = visible[(index + 1) % pair_count] if (closed or index + 1 < pair_count) else False
            if not previous_visible:
                add_cap(index, reverse=True)

            first = index * 6
            second = (index + 1) * 6
            for side in range(6):
                next_side = (side + 1) % 6
                add_quad(first + side, first + next_side, second + next_side, second + side, palette[side])

            if not next_visible:
                add_cap(index + 1, reverse=False)

        if not faces:
            return None
        return self.add_item(self.mesh_item(vertices, faces, color, alpha, draw_edges=False, face_colors=face_colors))

    def add_box(self, center, size, color="#ff8a3d", alpha=1.0, edge="#101820"):
        cx, cy, cz = center
        sx, sy, sz = size
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2
        vertexes = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        return self.add_item(self.mesh_item(vertexes, faces, color, alpha, edge))

    def add_cylinder(self, center, radius, height, color="#ff8a3d", alpha=1.0, segments=24):
        cx, cy, cz = center
        z0, z1 = cz - height / 2, cz + height / 2
        vertexes = [(cx, cy, z0), (cx, cy, z1)]
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertexes.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z0))
            vertexes.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z1))
        faces = []
        for i in range(segments):
            j = (i + 1) % segments
            b0, t0 = 2 + i * 2, 3 + i * 2
            b1, t1 = 2 + j * 2, 3 + j * 2
            faces.extend([(b0, b1, t1), (b0, t1, t0), (0, b0, b1), (1, t1, t0)])
        return self.add_item(self.mesh_item(vertexes, faces, color, alpha, edge="#0f1419"))

    def add_cone(self, tip, radius, height, color="#cbd5df", alpha=1.0, segments=24):
        tx, ty, tz = tip
        base_z = tz + height
        vertexes = [(tx, ty, tz), (tx, ty, base_z)]
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertexes.append((tx + radius * math.cos(angle), ty + radius * math.sin(angle), base_z))
        faces = []
        for i in range(segments):
            j = 2 + ((i + 1) % segments)
            k = 2 + i
            faces.append((0, k, j))
            faces.append((1, j, k))
        return self.add_item(self.mesh_item(vertexes, faces, color, alpha, edge="#687482"))

    def add_sphere_marker(self, position, size, color="#ff4d5d", alpha=0.9):
        if gl is None:
            return
        scatter = gl.GLScatterPlotItem(pos=np.asarray([position], dtype=np.float32), size=size, color=color_tuple(color, alpha), pxMode=True)
        self.add_item(scatter)

    def add_line(self, points, color="#ffffff", width=2.0, alpha=1.0, mode="line_strip"):
        if gl is None or len(points) < 2:
            return None
        item = gl.GLLinePlotItem(pos=np.asarray(points, dtype=np.float32), color=color_tuple(color, alpha), width=width, antialias=True, mode=mode)
        return self.add_item(item)

    def add_segment_strip(self, segment, progress=1.0, alpha=1.0):
        partial = segment.partial(progress) if progress < 0.999 else segment
        start = self.scene_point(partial.start)
        end = self.scene_point(partial.end)
        length = np.linalg.norm(end - start)
        width = self.scene_width(partial.width)
        height = self.scene_height(partial.height)
        if length < 0.025:
            self.add_sphere_marker(start, 7 + width * 3.6, partial.color, alpha)
            return
        direction_xy = end[:2] - start[:2]
        norm = np.linalg.norm(direction_xy)
        if norm < 0.001:
            perp = np.array((width / 2, 0.0, 0.0))
        else:
            perp_xy = np.array((-direction_xy[1], direction_xy[0])) / norm * (width / 2)
            perp = np.array((perp_xy[0], perp_xy[1], 0.0))
        vertical = np.array((0.0, 0.0, height / 2))
        vertexes = [
            start - perp - vertical,
            start + perp - vertical,
            end + perp - vertical,
            end - perp - vertical,
            start - perp + vertical,
            start + perp + vertical,
            end + perp + vertical,
            end - perp + vertical,
        ]
        faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        edge = "#a65228" if partial.segment_type != "support" else "#438eb8"
        self.add_item(self.mesh_item(vertexes, faces, partial.color, alpha, edge))

    def draw_bead_run(self, run):
        if not run:
            return
        has_flow_marks = any(segment.segment_type == "underextrusion" or float(segment.meta.get("flow_ratio", 0.0)) > 0.75 for segment in run)
        has_pressure_marks = any(segment.meta.get("profile") == "pressure" for segment in run)
        has_restart_marks = any(float(segment.meta.get("restart_gap_risk", 0.0)) > 0.0 or float(segment.meta.get("restart_blob_risk", 0.0)) > 0.0 for segment in run)
        max_step = 0.10 if (has_flow_marks or has_pressure_marks or has_restart_marks) else 0.16
        model_points, samples, total = self.generate_bead_samples_along_path(run, max_step=max_step)
        if len(model_points) < 2:
            return
        modifiers = self.apply_width_profile(samples, total)
        gaps = self.apply_gap_profile(samples, total)
        z_offsets = self.apply_z_offset_profile(samples)
        model_points = [
            np.asarray(point, dtype=float) + np.asarray((0.0, 0.0, z_offsets[index]), dtype=float)
            for index, point in enumerate(model_points)
        ]
        scene_points = [self.scene_point(point) for point in model_points]

        width = float(np.mean([self.scene_width(segment.width) for segment in run]))
        height = float(np.mean([self.scene_height(segment.height) for segment in run]))
        scene_points, modifiers = self.fillet_path(scene_points, modifiers, clamp(width * 0.92, 0.52, 2.35))
        segment_type = run[0].segment_type
        alpha = 0.92
        if segment_type == "support":
            alpha = 0.36
        elif segment_type == "underextrusion":
            alpha = 0.76
        elif segment_type == "ringing_offset":
            alpha = 0.88
        self.create_extrusion_bead_mesh(scene_points, width, height, run[0].color, width_modifiers=modifiers, gap_ranges=gaps, alpha=alpha)

    def draw_travel_stringing(self, segment, progress=1.0):
        risk = clamp(float(segment.meta.get("stringing_risk", 0.0)), 0.0, 1.0)
        if risk <= 0.04:
            return
        count = int(round(lerp(1, 6, risk)))
        count = max(1, min(6, count))
        start = np.asarray(segment.start, dtype=float)
        end = np.asarray(segment.end, dtype=float)
        layer_seed = int(segment.meta.get("layer_index", 0))
        for index in range(count):
            centered = index - (count - 1) / 2
            y_offset = centered * lerp(0.035, 0.075, risk)
            start_jitter = (self.deterministic_noise(index, layer_seed) - 0.5) * 0.025
            end_jitter = (self.deterministic_noise(index + 11, layer_seed) - 0.5) * 0.025
            sag = lerp(0.015, 0.100, risk) * (0.7 + 0.2 * abs(centered))
            curve = [
                start + np.asarray((0.00, y_offset + start_jitter, -0.15), dtype=float),
                (start + end) * 0.5 + np.asarray((0.0, y_offset * 0.35, -0.20 - sag), dtype=float),
                end + np.asarray((0.00, y_offset * 0.45 + end_jitter, -0.16), dtype=float),
            ]
            partial = partial_polyline3d(curve, progress)
            if len(partial) >= 2:
                alpha = 0.18 + risk * 0.42
                self.add_line([self.scene_point(point) for point in partial], "#f6ead0", 0.62, alpha)

    def draw_segment(self, segment, progress=1.0, active=False):
        if segment.segment_type == "travel":
            if active:
                partial = segment.partial(progress)
                self.add_line([self.scene_point(partial.start), self.scene_point(partial.end)], "#dfe7ef", 1.0, 0.60)
            self.draw_travel_stringing(segment, progress if active else 1.0)
            return
        if segment.segment_type == "stringing":
            self.add_line([self.scene_point(segment.start), self.scene_point(segment.end)], segment.color, 0.75, 0.22 + segment.defect_strength * 0.42)
            return
        if segment.segment_type == "blob":
            self.add_sphere_marker(self.scene_point(segment.start), 5.5 + segment.defect_strength * 11, segment.color, 0.68)
            return
        if segment.segment_type == "gap":
            if active:
                partial = segment.partial(progress)
                self.add_line([self.scene_point(partial.start), self.scene_point(partial.end)], "#ff4d5d", 1.2, 0.75)
            else:
                self.add_line([self.scene_point(segment.start), self.scene_point(segment.end)], "#ff4d5d", 1.0, 0.55)
            return
        if self.is_bead_segment(segment):
            partial = segment.partial(progress) if progress < 0.999 else segment
            self.draw_bead_run([partial])
            return
        self.add_segment_strip(segment, progress=progress, alpha=0.82)

    def add_nozzle(self, tip):
        tip = self.scene_point(tip) + np.array((0.0, 0.0, self.scene_height(self.engine.visual_segment_height()) * 0.72))
        ns = NOZZLE_SCALE
        self.add_cone(tip, radius=2.35 * ns, height=6.20 * ns, color="#dbe2ea", alpha=1.0)
        self.add_cylinder(center=(tip[0], tip[1], tip[2] + 0.42 * ns), radius=0.48 * ns, height=0.90 * ns, color="#29323a", alpha=1.0, segments=16)
        self.add_box(center=(tip[0], tip[1], tip[2] + 7.60 * ns), size=(6.40 * ns, 5.20 * ns, 3.20 * ns), color="#7f8c98", alpha=1.0, edge="#d7e0e8")
        self.add_box(center=(tip[0], tip[1], tip[2] + 5.45 * ns), size=(7.60 * ns, 1.40 * ns, 1.35 * ns), color="#ff8a3d", alpha=1.0, edge="#ffd0aa")
        self.add_line([(tip[0], tip[1], tip[2] + 9.00 * ns), (tip[0] - 4.8 * ns, tip[1] - 2.3 * ns, tip[2] + 15.5 * ns)], "#8190a0", 2.0, 0.75)
        self.add_line([(tip[0], tip[1], tip[2] + 9.00 * ns), (tip[0] + 4.8 * ns, tip[1] + 2.3 * ns, tip[2] + 15.5 * ns)], "#8190a0", 2.0, 0.75)

    def build_static_scene(self):
        self.add_grid()
        self.add_print_bed()

    def draw_completed_segments(self):
        for item in self.collect_bead_runs(self.engine.completed_segments):
            if isinstance(item, list):
                self.draw_bead_run(item)
            else:
                self.draw_segment(item, progress=1.0, active=False)

    def draw_active_segment_progress(self):
        if self.engine.active_segment is not None:
            self.draw_segment(self.engine.active_segment, progress=self.engine.active_segment_progress, active=True)

    def update_nozzle_pose(self):
        self.add_nozzle(self.engine.nozzle_position)

    def update_mode_scene(self, reset_camera=False):
        if self.view is None:
            return
        self.engine.ensure_current()
        if reset_camera or self.camera_mode != self.state.active_mode:
            self.setup_camera(self.state.active_mode)
        self.update_scene()

    def update_scene(self):
        if self.view is None:
            return
        self.engine.update_progress(self.state.animation_time)
        self.clear_scene()
        self.build_static_scene()
        self.draw_completed_segments()
        self.draw_active_segment_progress()
        self.update_nozzle_pose()

    def animate(self):
        self.update_scene()


class ParameterPanel(QWidget):
    parameters_changed = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(9)
        self.rebuild()

    def clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def rebuild(self):
        self.clear_layout()
        title = QLabel(MODE_LABELS[self.state.active_mode])
        title.setObjectName("PanelTitle")
        self.layout.addWidget(title)
        if self.state.active_mode == "intro":
            self.build_intro_panel()
        elif self.state.active_mode == "overhang":
            self.add_help("Açı, fan, hız ve support durumunun overhang sarkmasına etkisini gösterir.")
            self.build_overhang_controls()
        elif self.state.active_mode == "pressure":
            self.add_help("Pressure Advance köşe basıncı, blob ve gap davranışını temsili olarak gösterir.")
            self.build_pressure_controls()
        elif self.state.active_mode == "input":
            self.add_help("Shaper tipi ve hız/ivme değerleri yüzeydeki ringing izlerini değiştirir.")
            self.build_input_controls()
        elif self.state.active_mode == "retraction":
            self.add_help("Retraction travel sırasındaki sızıntıyı ve fazla retraction gap riskini etkiler.")
            self.build_retraction_controls()
        elif self.state.active_mode == "flow":
            self.add_help("Layer height × line width × speed hotend'in eritmesi gereken debiyi verir.")
            self.build_flow_controls()
        self.layout.addStretch(1)

    def build_intro_panel(self):
        self.add_help("Mouse ile 3D sahneyi döndürüp yakınlaştırabilirsin. Soldan bir mod seçerek ayarların etkisini izle.")
        cards = [
            ("FDM mantığı", "Filament eritilir ve nozzle üzerinden katman katman serilir."),
            ("3D temsil", "Sahne gerçek fizik çözmez; parametre etkisini teknik demo olarak gösterir."),
            ("Modlar", "Overhang, PA, ringing, stringing ve flow davranışları ayrı sahnelerde incelenir."),
        ]
        for heading, body in cards:
            card = QFrame()
            card.setObjectName("InfoCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 10, 12, 10)
            head = QLabel(heading)
            head.setObjectName("CardTitle")
            text = QLabel(body)
            text.setWordWrap(True)
            text.setObjectName("MutedText")
            layout.addWidget(head)
            layout.addWidget(text)
            self.layout.addWidget(card)

    def add_help(self, text):
        frame = QFrame()
        frame.setObjectName("HelpBox")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        label = QLabel("Bu mod ne gösterir?")
        label.setObjectName("CardTitle")
        body = QLabel(text)
        body.setWordWrap(True)
        body.setObjectName("MutedText")
        layout.addWidget(label)
        layout.addWidget(body)
        self.layout.addWidget(frame)

    def current_value(self, key, default):
        return self.state.current_params().get(key, default)

    def add_slider(self, label_text, key, min_value, max_value, step, unit, tooltip):
        frame = QFrame()
        frame.setObjectName("ControlRow")
        frame.setMinimumHeight(68)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        title = QLabel(label_text)
        title.setWordWrap(True)
        title.setToolTip(tooltip)
        value_label = QLabel()
        value_label.setObjectName("ValueLabel")
        value_label.setMinimumWidth(96)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(int(round((max_value - min_value) / step)))
        current = float(self.current_value(key, min_value))
        slider.setValue(int(round((current - min_value) / step)))
        slider.setToolTip(tooltip)

        def format_value(value):
            if step >= 1:
                return f"{int(round(value))} {unit}".strip()
            if step >= 0.1:
                return f"{value:.1f} {unit}".strip()
            return f"{value:.2f} {unit}".strip()

        def update_value(slider_value):
            value = clamp(min_value + slider_value * step, min_value, max_value)
            value = int(round(value)) if step >= 1 else round(value, 2)
            value_label.setText(format_value(value))
            self.state.update_parameter(key, value)
            self.parameters_changed.emit()

        value_label.setText(format_value(current))
        slider.valueChanged.connect(update_value)
        layout.addWidget(title, 0, 0)
        layout.addWidget(value_label, 0, 1)
        layout.addWidget(slider, 1, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        self.layout.addWidget(frame)

    def add_checkbox(self, label_text, key, tooltip):
        checkbox = QCheckBox(label_text)
        checkbox.setObjectName("ModernCheck")
        checkbox.setToolTip(tooltip)
        checkbox.setChecked(bool(self.current_value(key, False)))

        def changed(state):
            self.state.update_parameter(key, state == Qt.Checked.value)
            self.parameters_changed.emit()

        checkbox.stateChanged.connect(changed)
        self.layout.addWidget(checkbox)

    def add_combo(self, label_text, key, options, tooltip):
        frame = QFrame()
        frame.setObjectName("ControlRow")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        title = QLabel(label_text)
        combo = QComboBox()
        combo.addItems(options)
        combo.setToolTip(tooltip)
        current = str(self.current_value(key, options[0]))
        if current in options:
            combo.setCurrentText(current)

        def changed(text):
            self.state.update_parameter(key, text)
            self.parameters_changed.emit()

        combo.currentTextChanged.connect(changed)
        layout.addWidget(title)
        layout.addWidget(combo)
        self.layout.addWidget(frame)

    def build_overhang_controls(self):
        self.add_slider("Overhang açısı", "angle", 20, 80, 1, "derece", "Açı büyüdükçe sarkma riski artar.")
        self.add_slider("Fan hızı", "fan", 0, 100, 1, "%", "Plastiğin soğuma hızını temsil eder.")
        self.add_slider("Baskı hızı", "speed", 20, 120, 1, "mm/s", "Overhang bölgelerinde yüksek hız yerleşimi zorlaştırabilir.")
        self.add_checkbox("Support kullan", "support", "Desteksiz bölgenin altına geçici destek eklenmesini temsil eder.")

    def build_pressure_controls(self):
        self.add_slider("Pressure Advance", "pa", 0.00, 1.00, 0.01, "", "Köşe basıncını dengelemeyi temsil eder.")
        self.add_combo("Extruder tipi", "extruder", ["Direct Drive", "Bowden"], "Direct drive ve Bowden için ideal PA farklıdır.")
        self.add_slider("Baskı hızı", "speed", 30, 180, 1, "mm/s", "Hız arttıkça basınç değişimleri belirginleşir.")

    def build_input_controls(self):
        self.add_slider("Acceleration", "acceleration", 500, 10000, 100, "mm/s²", "Yüksek ivme ringing riskini artırabilir.")
        self.add_slider("Rezonans frekansı", "frequency", 20, 80, 1, "Hz", "Titreşime yatkın frekansı temsil eder.")
        self.add_combo("Shaper tipi", "shaper", ["Kapalı", "MZV", "EI", "2HUMP_EI"], "Input shaping titreşim izlerini azaltır.")
        self.add_slider("Baskı hızı", "speed", 40, 250, 1, "mm/s", "Yüksek hız titreşim etkisini görünür yapabilir.")

    def build_retraction_controls(self):
        self.add_slider("Retraction mesafesi", "retraction", 0.0, 8.0, 0.1, "mm", "Travel sırasında filamentin geri çekilmesini temsil eder.")
        self.add_slider("Nozzle sıcaklığı", "temperature", 180, 260, 1, "°C", "Sıcaklık arttıkça stringing riski artabilir.")
        self.add_slider("Travel hızı", "travel_speed", 50, 250, 1, "mm/s", "Hızlı travel sızıntı süresini azaltır.")
        self.add_combo("Extruder tipi", "extruder", ["Direct Drive", "Bowden"], "İdeal retraction mesafesi extruder tipine göre değişir.")

    def build_flow_controls(self):
        self.add_slider("Layer Height", "layer_height", 0.08, 0.40, 0.01, "mm", "Katman yüksekliği debiyi etkiler.")
        self.add_slider("Line Width", "line_width", 0.35, 0.80, 0.01, "mm", "Çizgi genişliği debiyi etkiler.")
        self.add_slider("Print Speed", "print_speed", 20, 250, 1, "mm/s", "Baskı hızı debiyi doğrudan artırır.")
        self.add_slider("Hotend kapasitesi", "max_flow", 4, 35, 1, "mm³/s", "Hotend'in eritebildiği yaklaşık hacimdir.")
        self.add_combo("Nozzle çapı", "nozzle_diameter", ["0.4", "0.6", "0.8"], "Nozzle çapı çizgi genişliği davranışını etkileyebilir.")


class InfoPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setObjectName("InfoPanel")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)
        self.score_label = QLabel()
        self.score_label.setObjectName("BigScore")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.calculated = QLabel()
        self.calculated.setObjectName("MetricBox")
        self.calculated.setWordWrap(True)
        self.bars_layout = QVBoxLayout()
        self.recommendation = QLabel()
        self.recommendation.setObjectName("MutedText")
        self.recommendation.setWordWrap(True)
        self.layout.addWidget(self.score_label)
        self.layout.addWidget(self.calculated)
        self.layout.addLayout(self.bars_layout)
        self.layout.addWidget(self.recommendation)
        self.layout.addStretch(1)
        self.update_info()

    def clear_bars(self):
        while self.bars_layout.count():
            item = self.bars_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def add_progress(self, label, value, is_risk):
        value = clamp(value, 0.0, 1.0)
        color = risk_color(value) if is_risk else quality_color(value)
        row = QFrame()
        row.setObjectName("ProgressCard")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(10, 7, 10, 7)
        title = QLabel(f"{label}: {int(round(value * 100))}/100")
        title.setObjectName("CardTitle")
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(round(value * 100)))
        bar.setTextVisible(False)
        bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color.name()}; border-radius: 4px; }}")
        layout.addWidget(title)
        layout.addWidget(bar)
        self.bars_layout.addWidget(row)

    def update_info(self):
        mode = self.state.active_mode
        params = self.state.current_params()
        score_label, normalized, score, is_risk = FDMModel.score_for_mode(mode, params)
        color = risk_color(normalized) if is_risk else quality_color(normalized)
        self.score_label.setText(f"{score_label}\n{score}/100")
        self.score_label.setStyleSheet(f"color: {color.name()};")
        self.calculated.setText(f"Hesaplanan değer\n{FDMModel.calculated_value_text(mode, params)}")
        self.recommendation.setText(f"Tavsiye: {FDMModel.recommendation_text(mode, params)}")
        self.clear_bars()
        for label, value, bar_is_risk in FDMModel.progress_items(mode, params):
            self.add_progress(label, value, bar_is_risk)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = SimulationState()
        self._updating_preset = False
        self.setWindowTitle("FDM Parametreleri Görselleştiricisi")
        self.resize(1600, 900)
        self.setMinimumSize(1240, 760)
        self.apply_theme()
        self.build_ui()
        self.build_menu()
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 14)
        root_layout.setSpacing(12)
        root_layout.addWidget(self.build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.build_left_panel())
        self.scene_widget = GLSceneWidget(self.state)
        splitter.addWidget(self.scene_widget)
        splitter.addWidget(self.build_right_panel())
        splitter.setSizes([236, 900, 430])
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self.build_footer())
        self.setCentralWidget(root)

    def build_header(self):
        frame = QFrame()
        frame.setObjectName("Header")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)
        title = QLabel("FDM Parametreleri Görselleştiricisi")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Yeni Başlayanlar İçin FDM 3D Baskı Ayarları Görsel Rehberi")
        subtitle.setObjectName("AppSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame

    def build_left_panel(self):
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(236)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self.panel_title("Modlar"))
        self.mode_buttons = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for mode, label in MODES:
            button = QPushButton(label)
            button.setObjectName("ModeButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, selected_mode=mode: self.set_mode(selected_mode))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            layout.addWidget(button)
        self.mode_buttons[self.state.active_mode].setChecked(True)
        layout.addWidget(self.separator())
        layout.addWidget(self.panel_title("Animasyon"))
        for text, slot in [
            ("Başlat", self.start_animation),
            ("Durdur", self.stop_animation),
            ("Sıfırla", self.reset_animation),
            ("Yavaşlat", self.slower_animation),
            ("Hızlandır", self.faster_animation),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Animasyon hızı"))
        speed_row.addStretch(1)
        self.speed_value = QLabel("1.00x")
        self.speed_value.setObjectName("ValueLabel")
        speed_row.addWidget(self.speed_value)
        layout.addLayout(speed_row)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(20)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.set_animation_speed_from_slider)
        layout.addWidget(self.speed_slider)

        layout.addWidget(self.separator())
        layout.addWidget(self.panel_title("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["PLA", "PETG", "ABS", "Custom"])
        self.preset_combo.setCurrentText(self.state.selected_preset)
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        layout.addWidget(self.preset_combo)
        export_button = QPushButton("Sahneyi PNG olarak kaydet")
        export_button.clicked.connect(self.export_png)
        layout.addWidget(export_button)
        layout.addStretch(1)
        return panel

    def build_right_panel(self):
        panel = QFrame()
        panel.setObjectName("RightPanel")
        panel.setMinimumWidth(390)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.parameter_panel = ParameterPanel(self.state)
        self.parameter_panel.parameters_changed.connect(self.parameters_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.parameter_panel)
        layout.addWidget(scroll, 4)
        self.info_panel = InfoPanel(self.state)
        layout.addWidget(self.info_panel, 1)
        return panel

    def build_footer(self):
        footer = QFrame()
        footer.setObjectName("Footer")
        layout = QGridLayout(footer)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setHorizontalSpacing(16)
        self.footer_warning = QLabel("Bu uygulama gerçek fizik simülasyonu değildir; öğretici ve temsili 3D görselleştirme yapar.")
        self.footer_warning.setObjectName("WarningText")
        self.footer_warning.setWordWrap(True)
        self.footer_explanation = QLabel()
        self.footer_explanation.setObjectName("BodyText")
        self.footer_explanation.setWordWrap(True)
        self.copy_button = QPushButton("Rapor İçin Değerleri Kopyala")
        self.copy_button.clicked.connect(self.copy_report)
        layout.addWidget(self.footer_warning, 0, 0)
        layout.addWidget(self.footer_explanation, 1, 0)
        layout.addWidget(self.copy_button, 0, 1, 2, 1)
        layout.setColumnStretch(0, 1)
        self.update_footer()
        return footer

    def build_menu(self):
        file_menu = self.menuBar().addMenu("Dosya")
        export_action = QAction("Sahneyi PNG olarak kaydet", self)
        export_action.triggered.connect(self.export_png)
        file_menu.addAction(export_action)

    def panel_title(self, text):
        label = QLabel(text)
        label.setObjectName("PanelTitle")
        return label

    def separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("Separator")
        return line

    def set_mode(self, mode):
        self.state.set_mode(mode)
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
        self.refresh_panels(rebuild_parameters=True)

    def start_animation(self):
        self.state.running = True

    def stop_animation(self):
        self.state.running = False

    def reset_animation(self):
        self.state.reset()
        self.scene_widget.engine.reset()
        self.scene_widget.update_mode_scene(reset_camera=False)

    def slower_animation(self):
        self.speed_slider.setValue(max(self.speed_slider.minimum(), self.speed_slider.value() - 20))

    def faster_animation(self):
        self.speed_slider.setValue(min(self.speed_slider.maximum(), self.speed_slider.value() + 20))

    def set_animation_speed_from_slider(self, slider_value):
        self.state.animation_speed = slider_value / 100.0
        self.speed_value.setText(f"{self.state.animation_speed:.2f}x")

    def apply_preset(self, preset_name):
        self._updating_preset = True
        self.state.apply_preset(preset_name)
        self.state.reset()
        self.refresh_panels(rebuild_parameters=True)
        self._updating_preset = False

    def parameters_changed(self):
        if not self._updating_preset and self.state.selected_preset != "Custom":
            self.state.selected_preset = "Custom"
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
        self.state.reset()
        self.refresh_panels(rebuild_parameters=False)

    def refresh_panels(self, rebuild_parameters=False):
        if rebuild_parameters:
            self.parameter_panel.rebuild()
        self.info_panel.update_info()
        self.update_footer()
        self.scene_widget.update_mode_scene(reset_camera=rebuild_parameters)

    def update_footer(self):
        self.footer_explanation.setText(FDMModel.explanation_text(self.state.active_mode, self.state.current_params()))

    def tick(self):
        if self.state.running:
            self.state.animation_time = (self.state.animation_time + 0.010 * self.state.animation_speed) % 1000.0
            self.scene_widget.animate()

    def export_png(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Sahneyi PNG olarak kaydet", "fdm-3d-sahne.png", "PNG Image (*.png)")
        if not filename:
            return
        if not filename.lower().endswith(".png"):
            filename += ".png"
        pixmap = self.scene_widget.grab()
        if pixmap.isNull():
            pixmap = QPixmap(self.size())
            self.render(pixmap)
        if pixmap.save(filename, "PNG"):
            QMessageBox.information(self, "PNG kaydedildi", f"Sahne kaydedildi:\n{filename}")
        else:
            QMessageBox.warning(self, "PNG kaydedilemedi", "Seçilen konuma PNG dosyası kaydedilemedi.")

    def copy_report(self):
        text = FDMModel.report_copy_text(self.state.active_mode, self.state.current_params())
        QApplication.clipboard().setText(text)
        old_text = self.copy_button.text()
        self.copy_button.setText("Kopyalandı")
        QTimer.singleShot(1200, lambda: self.copy_button.setText(old_text))

    def apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #090d13; color: #eaf0f6; }
            QWidget { color: #eaf0f6; font-family: Segoe UI, Arial; font-size: 10pt; }
            QMenuBar { background: #0f151d; color: #dce6ee; border-bottom: 1px solid #1d2a36; }
            QMenuBar::item:selected, QMenu::item:selected { background: #263748; }
            QMenu { background: #111923; color: #eaf0f6; border: 1px solid #273746; }
            #Header, #SidePanel, #RightPanel, #Footer, #InfoPanel, #GLSceneShell {
                background: #101821; border: 1px solid #1f2c39; border-radius: 8px;
            }
            #Header { background: #111b25; }
            #AppTitle { font-size: 20pt; font-weight: 700; color: #ffffff; }
            #AppSubtitle { color: #9fb0c0; }
            #PanelTitle { font-size: 11pt; font-weight: 700; color: #ffffff; }
            #CardTitle { font-weight: 700; color: #f3f7fb; }
            #MutedText { color: #aebaca; }
            #BodyText { color: #dce6ee; }
            #WarningText { color: #ffcf7a; font-weight: 600; }
            #InfoCard, #HelpBox, #ControlRow, #ProgressCard, #MetricBox {
                background: #121d27; border: 1px solid #263645; border-radius: 8px;
            }
            #MetricBox, #FormulaText {
                padding: 7px; color: #cfe8ff; background: #0d141c; border: 1px solid #253546; border-radius: 8px;
            }
            #BigScore {
                font-size: 15pt; font-weight: 800; background: #0d141c; border: 1px solid #253546;
                border-radius: 8px; padding: 8px;
            }
            #ErrorBox {
                color: #ffb3b8; background: #27151a; border: 1px solid #63303b; border-radius: 8px; padding: 14px;
            }
            QPushButton {
                background: #1a2633; color: #eaf0f6; border: 1px solid #2e4254; border-radius: 7px; padding: 8px 9px;
            }
            QPushButton:hover { background: #233345; border-color: #436176; }
            QPushButton:pressed { background: #16212c; }
            QPushButton#ModeButton { text-align: left; }
            QPushButton#ModeButton:checked { background: #25384a; border-color: #ff8a3d; color: #ffffff; }
            QComboBox {
                background: #0d141c; color: #eaf0f6; border: 1px solid #2d4153; border-radius: 6px; padding: 7px 8px;
            }
            QComboBox QAbstractItemView { background: #101821; color: #eaf0f6; selection-background-color: #25384a; }
            QSlider::groove:horizontal { height: 6px; background: #263645; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #ff8a3d; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 16px; height: 16px; margin: -6px 0; background: #f5f8fb;
                border: 2px solid #ff8a3d; border-radius: 8px;
            }
            QCheckBox {
                background: #121d27; border: 1px solid #263645; border-radius: 8px; padding: 8px;
            }
            QProgressBar {
                background: #263645; border: 0; border-radius: 4px; height: 8px;
            }
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { background: #0c1219; width: 10px; margin: 2px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #2b3c4d; min-height: 36px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            #Separator { color: #263645; background: #263645; }
            #ValueLabel { color: #ffbc84; font-weight: 700; }
            """
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FDM Parametreleri Görselleştiricisi")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
