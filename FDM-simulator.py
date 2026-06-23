import math
import sys
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QAction, QColor, QPixmap, QSurfaceFormat
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
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    GL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency fallback
    pg = None
    gl = None
    PGVector = None
    GL_IMPORT_ERROR = exc

# modların iç keyleri sabit kalsın, ekranda görünen isim ayrı iş
MODE_KEYS = ("intro", "overhang", "pressure", "input", "retraction", "flow")
DEFAULT_TERM_MODE = "Açıklamalı"
TERM_MODE_LABELS = {
    "Açıklamalı": {
        "intro": "Ana Sayfa",
        "overhang": "Köprüleme / Soğutma",
        "pressure": "Basınç Dengeleme (Pressure Advance)",
        "input": "Titreşim Sönümleme / Ringing",
        "retraction": "Geri Çekme / İpliklenme",
        "flow": "Hacimsel Debi",
    },
    "Teknik": {
        "intro": "Home",
        "overhang": "Bridge / Cooling",
        "pressure": "Pressure Advance",
        "input": "Input Shaping / Ringing",
        "retraction": "Retraction / Stringing",
        "flow": "Volumetric Flow",
    },
}

MODES = [(mode, TERM_MODE_LABELS[DEFAULT_TERM_MODE][mode]) for mode in MODE_KEYS]
MODE_LABELS = dict(MODES)

# sol panel dar, bazı isimleri burada elle iki satıra kırıyoruz
MODE_BUTTON_BREAKS = {
    "Açıklamalı": {
        "overhang": "Köprüleme /\nSoğutma",
        "pressure": "Basınç Dengeleme\n(Pressure Advance)",
        "input": "Titreşim Sönümleme\n/ Ringing",
        "retraction": "Geri Çekme\n/ İpliklenme",
    },
    "Teknik": {
        "overhang": "Bridge /\nCooling",
        "input": "Input Shaping\n/ Ringing",
        "retraction": "Retraction\n/ Stringing",
    },
}

# içeride custom kalsın, kullanıcıya özel görünsün yeter
PRESET_DISPLAY_NAMES = {"PLA": "PLA", "PETG": "PETG", "ABS": "ABS", "Custom": "Özel"}
PRESET_DISPLAY_TO_VALUE = {display: value for value, display in PRESET_DISPLAY_NAMES.items()}


def mode_label(mode, term_mode=DEFAULT_TERM_MODE):
    labels = TERM_MODE_LABELS.get(term_mode, TERM_MODE_LABELS[DEFAULT_TERM_MODE])
    return labels.get(mode, MODE_LABELS.get(mode, mode))


def mode_button_label(mode, term_mode=DEFAULT_TERM_MODE):
    return MODE_BUTTON_BREAKS.get(term_mode, {}).get(mode, mode_label(mode, term_mode))


def preset_display_name(preset_name):
    return PRESET_DISPLAY_NAMES.get(preset_name, preset_name)


def preset_internal_name(display_name):
    return PRESET_DISPLAY_TO_VALUE.get(display_name, display_name)

# ilk açılış değerleri, preset seçilince bunların üstüne yazılıyor
DEFAULT_PARAMETERS = {
    "intro": {},
    "overhang": {"angle": 40, "fan": 70, "speed": 55, "support": False},
    "pressure": {"pa": 0.05, "extruder": "Direct Drive", "speed": 80},
    "input": {"acceleration": 3500, "frequency": 45, "shaper": "MZV", "speed": 100},
    "retraction": {"retraction": 1.0, "temperature": 205, "travel_speed": 160, "extruder": "Direct Drive"},
    "flow": {"layer_height": 0.20, "line_width": 0.45, "print_speed": 70, "max_flow": 12, "nozzle_diameter": "0.4"},
}

PRESET_PARAMETERS = {
    "PLA": {
        "overhang": {"angle": 30, "fan": 95, "speed": 55, "support": False},
        "pressure": {"pa": 0.05, "extruder": "Direct Drive", "speed": 80},
        "input": {"acceleration": 3500, "frequency": 45, "shaper": "MZV", "speed": 100},
        "retraction": {"retraction": 1.0, "temperature": 205, "travel_speed": 170, "extruder": "Direct Drive"},
        "flow": {"layer_height": 0.20, "line_width": 0.45, "print_speed": 70, "max_flow": 12, "nozzle_diameter": "0.4"},
    },
    "PETG": {
        "overhang": {"angle": 45, "fan": 45, "speed": 45, "support": False},
        "pressure": {"pa": 0.08, "extruder": "Direct Drive", "speed": 70},
        "input": {"acceleration": 3000, "frequency": 42, "shaper": "MZV", "speed": 90},
        "retraction": {"retraction": 1.4, "temperature": 240, "travel_speed": 150, "extruder": "Direct Drive"},
        "flow": {"layer_height": 0.20, "line_width": 0.45, "print_speed": 55, "max_flow": 10, "nozzle_diameter": "0.4"},
    },
    "ABS": {
        "overhang": {"angle": 40, "fan": 20, "speed": 45, "support": True},
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
BEAD_CROSS_SECTION_SEGMENTS = 14
ROUND_GEOMETRY_SEGMENTS = 48
BEAD_SAMPLE_STEP = 0.10
RETRACTION_TOWER_POINTS = 48
FLOW_REASONABLE_LINE_RATIO_MIN = 0.90
FLOW_REASONABLE_LINE_RATIO_MAX = 1.40
PA_SLIDER_MIN = 0.0
PA_SLIDER_MAX = 0.20
PA_SLIDER_STEP = 0.005
PRESSURE_ADVANCE_SETTINGS = {
    "Direct Drive": {"ideal": 0.05, "tolerance": 0.035},
    "Bowden": {"ideal": 0.13, "tolerance": 0.060},
}
PA_DEFECT_SPAN_MULTIPLIER = 3.2
RETRACTION_SETTINGS = {
    "Direct Drive": {"ideal": 1.0, "safe_min": 0.7, "safe_max": 1.5, "over_span": 2.4},
    "Bowden": {"ideal": 4.5, "safe_min": 3.5, "safe_max": 5.5, "over_span": 3.2},
}


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * clamp(t, 0.0, 1.0))


def smoothstep(value):
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


# bridge tarafında sahne ve risk aynı küçük paketten beslensin diye
def bridge_scene_config(params):
    span_mm = float(params.get("angle", 40))
    fan = float(params.get("fan", 70))
    speed = float(params.get("speed", 55))
    support = bool(params.get("support", False))

    span_visual = clamp((span_mm - 10) / 70, 0.0, 1.0)
    span_factor = clamp((span_mm - 20) / 60, 0.0, 1.0)
    fan_factor = clamp(1.0 - fan / 100, 0.0, 1.0)
    speed_factor = clamp((speed - 40) / 80, 0.0, 1.0)
    support_factor = 0.25 if support else 1.0
    sag_strength = clamp((0.60 * span_factor + 0.25 * fan_factor + 0.15 * speed_factor) * support_factor, 0.0, 1.0)

    pillar_width = 0.94
    gap_width = lerp(1.28, 3.20, span_visual)
    pillar_center_offset = gap_width / 2 + pillar_width / 2
    bridge_overlap = 0.12

    return {
        "span_mm": span_mm,
        "span_factor": span_factor,
        "fan_factor": fan_factor,
        "speed_factor": speed_factor,
        "support_factor": support_factor,
        "sag_strength": sag_strength,
        "pillar_layers": 7,
        "bridge_layers": 7,
        "left_pillar_x": -pillar_center_offset,
        "right_pillar_x": pillar_center_offset,
        "pillar_width": pillar_width,
        "pillar_depth": 1.56,
        "bridge_start_x": -gap_width / 2 - bridge_overlap,
        "bridge_end_x": gap_width / 2 + bridge_overlap,
        "bridge_y_positions": np.linspace(-0.54, 0.54, 5),
        "support_x_positions": np.linspace(-gap_width * 0.26, gap_width * 0.26, 3),
    }


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
        # parametre değişmediyse katmanları tekrar üretmeye gerek yok
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
        # animasyon her küçük çizgiyi sırayla geziyor, süreleri burada çıkıyor
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
        # mod keyleri burada dağılıyor, görünen isimlerle işi yok
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
            layers.append(layer)
        return layers

    def generate_overhang(self, params):
        # köprü çizgileri alt kulelerin üstüne katman katman kuruluyor
        config = bridge_scene_config(params)
        risk = config["sag_strength"]
        layers = []
        total_layers = config["pillar_layers"] + config["bridge_layers"]
        for layer_index in range(total_layers):
            z = self.visual_layer_z(layer_index)
            height = self.visual_segment_height()
            layer = ToolpathLayer(z, layer_index)
            self.add_rectangle_layer(layer, config["left_pillar_x"], 0, z, config["pillar_width"], config["pillar_depth"], width=0.34, height=height)
            self.add_rectangle_layer(layer, config["right_pillar_x"], 0, z, config["pillar_width"], config["pillar_depth"], width=0.34, height=height)

            if layer_index >= config["pillar_layers"]:
                bridge_index = layer_index - config["pillar_layers"]
                layer_factor = (bridge_index + 1) / config["bridge_layers"]
                bridge_width = lerp(0.24, 0.30, config["span_factor"])
                for line_index, y in enumerate(config["bridge_y_positions"]):
                    start = (config["bridge_start_x"], float(y), z)
                    end = (config["bridge_end_x"], float(y), z)
                    if (bridge_index + line_index) % 2:
                        start, end = end, start
                    layer.segment_list.append(
                        ToolpathSegment(
                            start,
                            end,
                            bridge_width,
                            height,
                            "extrusion",
                            "#ff8a3d",
                            defect_strength=risk,
                            meta={
                                "bridge_profile": True,
                                "overhang_risk": risk,
                                "bridge_sag_strength": risk,
                                "bridge_anchor_left": config["bridge_start_x"],
                                "bridge_anchor_right": config["bridge_end_x"],
                                "bridge_span_mm": config["span_mm"],
                                "overhang_layer_factor": layer_factor,
                                "overhang_angle_factor": config["span_factor"],
                                "overhang_cooling_factor": config["fan_factor"],
                                "overhang_speed_factor": config["speed_factor"],
                                "overhang_support_factor": config["support_factor"],
                            },
                        )
                    )
            layers.append(layer)
        return layers

    def generate_pressure(self, params):
        # pa hatasını köşelerde şişme ya da incelme olarak işaretliyoruz
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
            pressure_corners = corners[:-1]
            for side_index, (start, end) in enumerate(zip(corners[:-1], corners[1:])):
                meta = {
                    "profile": "pressure",
                    "pa_low": low,
                    "pa_high": high,
                    "pa_corners": pressure_corners,
                    "pa_corner_radius": 0.42,
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
        # gerçek titreşim simülasyonu değil, okunur bir ringing dalgası
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
        # iki kule arası boşta hareket, ipliklenme için iyi bir sahne veriyor
        result = FDMModel.retraction_stringing_risk(params)
        stringing = result["stringing_risk"]
        gap = result["restart_gap_risk"]
        layers = []
        centers = [(-1.8, 0.0), (1.8, 0.0)]
        for layer_index in range(16):
            z = self.visual_layer_z(layer_index)
            height = self.visual_segment_height()
            layer = ToolpathLayer(z, layer_index)
            next_z = self.visual_layer_z(layer_index + 1) if layer_index < 15 else z + self.default_layer_height * self.visual_z_scale
            travel_z = z + 0.24
            next_travel_z = next_z + 0.24
            radius = 0.36
            tower_a_seam = (centers[0][0] + radius, centers[0][1], z)
            tower_b_seam = (centers[1][0] - radius, centers[1][1], z)
            tower_a_travel = (tower_a_seam[0], tower_a_seam[1], travel_z)
            tower_b_travel = (tower_b_seam[0], tower_b_seam[1], travel_z)
            tower_a_next_travel = (tower_a_seam[0], tower_a_seam[1], next_travel_z)
            start_blob = clamp(stringing - 0.20, 0.0, 1.0)
            self.add_circle_layer(
                layer,
                centers[0][0],
                centers[0][1],
                z,
                radius=radius,
                width=0.20,
                height=height,
                points=RETRACTION_TOWER_POINTS,
                start_angle=0.0,
                meta={
                    "tower": "first",
                    "layer_index": layer_index,
                    "restart_gap_risk": gap if layer_index > 0 else 0.0,
                    "restart_blob_risk": start_blob if layer_index > 0 else 0.0,
                    "restart_total_length": 2 * math.pi * radius,
                },
            )
            travel = ToolpathSegment(
                tower_a_travel,
                tower_b_travel,
                0.05,
                0.03,
                "travel",
                "#dfe7ef",
                defect_strength=stringing,
                meta={"stringing_risk": stringing, "layer_index": layer_index, "travel_direction": "A_to_B"},
            )
            layer.segment_list.append(travel)
            self.add_circle_layer(
                layer,
                centers[1][0],
                centers[1][1],
                z,
                radius=radius,
                width=0.20,
                height=height,
                points=RETRACTION_TOWER_POINTS,
                start_angle=math.pi,
                meta={
                    "tower": "second",
                    "layer_index": layer_index,
                    "restart_gap_risk": gap,
                    "restart_blob_risk": start_blob,
                    "restart_total_length": 2 * math.pi * radius,
                },
            )
            return_travel = ToolpathSegment(
                tower_b_travel,
                tower_a_next_travel,
                0.05,
                0.03,
                "travel",
                "#dfe7ef",
                defect_strength=stringing,
                meta={"stringing_risk": stringing * 0.85, "layer_index": layer_index, "travel_direction": "B_to_A"},
            )
            layer.segment_list.append(return_travel)
            layers.append(layer)
        return layers

    def generate_flow(self, params):
        # flow limiti aşılırsa çizgi incelmesi ve boşluk buradan besleniyor
        result = FDMModel.volumetric_flow_risk(params)
        risk = result["risk"]
        ratio = result["ratio"]
        line_width = float(params.get("line_width", 0.45))
        layer_height = float(params.get("layer_height", 0.20))
        nozzle_diameter = result["nozzle_diameter"]
        line_to_nozzle_ratio = result["line_to_nozzle_ratio"]
        nozzle_visual_factor = lerp(0.92, 1.24, clamp((nozzle_diameter - 0.4) / 0.4, 0.0, 1.0))
        line_visual_width = lerp(0.18, 0.34, clamp((line_width - 0.32) / 0.52, 0, 1))
        flow_load = smoothstep(clamp((ratio - 0.75) / 0.75, 0.0, 1.0))
        visual_width = line_visual_width * nozzle_visual_factor * lerp(1.0, 0.74, flow_load)
        flow_meta = {
            "flow_ratio": ratio,
            "flow_risk": risk,
            "nozzle_diameter": nozzle_diameter,
            "line_to_nozzle_ratio": line_to_nozzle_ratio,
        }
        layers = []
        for layer_index in range(10):
            z = self.visual_layer_z(layer_index, layer_height)
            height = self.visual_segment_height(layer_height)
            layer = ToolpathLayer(z, layer_index)
            for line_index, y in enumerate(np.linspace(-0.82, 0.82, 5)):
                start_x = -2.8
                end_x = 2.8
                meta = dict(flow_meta, line_index=line_index, layer_index=layer_index)
                if ratio <= 1.0:
                    layer.segment_list.append(ToolpathSegment((start_x, y, z), (end_x, y, z), visual_width, height, "extrusion", "#ff8a3d", defect_strength=risk, meta=meta))
                else:
                    layer.segment_list.append(ToolpathSegment((start_x, y, z), (end_x, y, z), visual_width, height * 0.78, "underextrusion", "#ff8a3d", defect_strength=risk, meta=meta))
            layers.append(layer)
        return layers


class FDMModel:
    """Temsili hesaplamalar ve yeni başlayan dostu açıklamalar."""

    @staticmethod
    def overhang_risk(params):
        return bridge_scene_config(params)["sag_strength"]

    @staticmethod
    def pressure_advance_quality(params):
        # burada kalite skoru var, risk skoru gibi ters okunmasın
        pa = clamp(float(params.get("pa", 0.05)), PA_SLIDER_MIN, PA_SLIDER_MAX)
        extruder = params.get("extruder", "Direct Drive")
        speed = float(params.get("speed", 80))
        profile = PRESSURE_ADVANCE_SETTINGS.get(extruder, PRESSURE_ADVANCE_SETTINGS["Direct Drive"])
        ideal = profile["ideal"]
        tolerance = profile["tolerance"]
        speed_factor = clamp((speed - 30) / (180 - 30), 0.0, 1.0)
        effective_tolerance = tolerance * (1.0 - 0.35 * speed_factor)
        defect_amplifier = 1.0 + 1.05 * speed_factor
        distance = abs(pa - ideal)
        quality = clamp(1.0 - (distance / max(effective_tolerance, 0.001)) ** 1.4, 0.0, 1.0)
        defect_span = max(effective_tolerance * PA_DEFECT_SPAN_MULTIPLIER, 0.001)
        base_low_pa_defect = smoothstep(clamp((ideal - pa) / defect_span, 0.0, 1.0))
        base_high_pa_defect = smoothstep(clamp((pa - ideal) / defect_span, 0.0, 1.0))
        low_pa_defect = clamp(base_low_pa_defect * defect_amplifier, 0.0, 1.0)
        high_pa_defect = clamp(base_high_pa_defect * defect_amplifier, 0.0, 1.0)
        return {
            "quality": quality,
            "low_pa_defect": low_pa_defect,
            "high_pa_defect": high_pa_defect,
            "ideal": ideal,
            "effective_tolerance": effective_tolerance,
            "pa": pa,
            "speed": speed,
            "speed_factor": speed_factor,
            "defect_amplifier": defect_amplifier,
        }

    @staticmethod
    def input_shaping_risk(params):
        # shaper kapalıysa risk tam geliyor, seçilince çarpanla kısılıyor
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
        # az geri çekme ipliklenme, fazla geri çekme restart boşluğu gibi düşün
        retraction = float(params.get("retraction", 1.0))
        temperature = float(params.get("temperature", 205))
        travel_speed = float(params.get("travel_speed", 160))
        extruder = params.get("extruder", "Direct Drive")
        profile = RETRACTION_SETTINGS.get(extruder, RETRACTION_SETTINGS["Direct Drive"])
        ideal = profile["ideal"]
        safe_min = profile["safe_min"]
        safe_max = profile["safe_max"]
        under_retract = smoothstep(clamp((safe_min - retraction) / max(safe_min, 0.001), 0.0, 1.0))
        over_retract = smoothstep(clamp((retraction - safe_max) / max(profile["over_span"], 0.001), 0.0, 1.0))
        temp_factor = smoothstep(clamp((temperature - 215) / 45, 0.0, 1.0))
        travel_factor = smoothstep(clamp((170 - travel_speed) / 120, 0.0, 1.0))
        base_stringing = 0.72 * under_retract + 0.18 * temp_factor + 0.10 * travel_factor
        stringing_risk = clamp(base_stringing * (1.0 - 0.68 * over_retract), 0.0, 1.0)
        restart_gap_risk = clamp(over_retract, 0, 1)
        combined_risk = max(stringing_risk, restart_gap_risk)
        return {
            "stringing_risk": stringing_risk,
            "restart_gap_risk": restart_gap_risk,
            "combined_risk": combined_risk,
            "ideal": ideal,
            "safe_min": safe_min,
            "safe_max": safe_max,
            "under_retract": under_retract,
            "over_retract": over_retract,
        }

    @staticmethod
    def volumetric_flow_risk(params):
        # debi hesabı basit kalsın, nozzle sadece yorum tarafına yardım ediyor
        layer_height = float(params.get("layer_height", 0.20))
        line_width = float(params.get("line_width", 0.45))
        print_speed = float(params.get("print_speed", 70))
        max_flow = float(params.get("max_flow", 12))
        nozzle_diameter = float(params.get("nozzle_diameter", 0.4))
        flow = layer_height * line_width * print_speed
        ratio = flow / max(max_flow, 0.001)
        line_to_nozzle_ratio = line_width / max(nozzle_diameter, 0.001)
        low_line_ratio_risk = smoothstep(clamp((FLOW_REASONABLE_LINE_RATIO_MIN - line_to_nozzle_ratio) / 0.28, 0.0, 1.0)) * 0.10
        high_line_ratio_risk = smoothstep(clamp((line_to_nozzle_ratio - FLOW_REASONABLE_LINE_RATIO_MAX) / 0.45, 0.0, 1.0)) * 0.22
        flow_risk = smoothstep(clamp((ratio - 0.75) / 0.55, 0, 1))
        risk = clamp(flow_risk + low_line_ratio_risk + high_line_ratio_risk, 0, 1)
        return {
            "flow": flow,
            "ratio": ratio,
            "risk": risk,
            "flow_risk": flow_risk,
            "nozzle_diameter": nozzle_diameter,
            "line_to_nozzle_ratio": line_to_nozzle_ratio,
            "line_ratio_risk": clamp(low_line_ratio_risk + high_line_ratio_risk, 0, 1),
        }

    @staticmethod
    def score_for_mode(mode, params):
        # pressure kalite, diğer modlar risk gibi okunuyor
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
            return "Durum", 0.0, 0, False
        return "Risk Skoru", risk, int(round(risk * 100)), True

    @staticmethod
    def score_level_text(score, is_risk):
        if is_risk:
            if score <= 33:
                return "Düşük"
            if score <= 66:
                return "Orta"
            return "Yüksek"
        if score >= 80:
            return "İyi"
        if score >= 50:
            return "Orta"
        return "Zayıf"

    @staticmethod
    def score_display_text(mode, score_label, score, is_risk):
        if mode == "intro":
            return "Durum: Simülasyon hazır"
        return f"{score_label}: {score}/100 - {FDMModel.score_level_text(score, is_risk)}"

    @staticmethod
    def formatted_parameter_lines(mode, params):
        # rapora ham değişken adı basmayalım diye küçük çeviri tablosu
        specs = {
            "overhang": [
                ("angle", "Köprü açıklığı", "mm"),
                ("fan", "Fan hızı", "%"),
                ("speed", "Baskı hızı", "mm/s"),
                ("support", "Destek kullan", ""),
            ],
            "pressure": [
                ("pa", "Pressure Advance", ""),
                ("extruder", "Ekstruder tipi", ""),
                ("speed", "Test hızı", "mm/s"),
            ],
            "input": [
                ("acceleration", "İvme", "mm/s²"),
                ("frequency", "Rezonans frekansı", "Hz"),
                ("shaper", "Shaper tipi", ""),
                ("speed", "Baskı hızı", "mm/s"),
            ],
            "retraction": [
                ("retraction", "Geri çekme mesafesi", "mm"),
                ("temperature", "Nozzle sıcaklığı", "°C"),
                ("travel_speed", "Boşta hareket hızı (Travel)", "mm/s"),
                ("extruder", "Ekstruder tipi", ""),
            ],
            "flow": [
                ("layer_height", "Katman yüksekliği", "mm"),
                ("line_width", "Çizgi genişliği", "mm"),
                ("print_speed", "Baskı hızı", "mm/s"),
                ("max_flow", "Hotend kapasitesi", "mm³/s"),
                ("nozzle_diameter", "Nozzle çapı", "mm"),
            ],
        }

        def format_value(value, unit):
            if isinstance(value, bool):
                return "Evet" if value else "Hayır"
            if isinstance(value, float):
                text = f"{value:.3f}" if abs(value) < 0.01 else f"{value:.2f}"
                text = text.rstrip("0").rstrip(".")
            else:
                text = str(value)
            return f"{text} {unit}".strip()

        lines = []
        for key, label, unit in specs.get(mode, []):
            if key in params:
                lines.append(f"- {label}: {format_value(params[key], unit)}")
        if not lines:
            lines.append("- Parametre yok")
        return lines

    @staticmethod
    def explanation_text(mode, params):
        if mode == "intro":
            return "FDM baskıda filament eritilir, nozzle'dan çıkarılır ve katman katman model oluşturulur."
        if mode == "overhang":
            risk = FDMModel.overhang_risk(params)
            if risk < 0.34:
                return "Köprüleme temiz görünüyor; iki destek kulesi arasındaki filament hatları düz kalıyor."
            if risk < 0.67:
                return "Köprü ortasında sarkma artıyor; fan ve baskı hızı etkisi sahnede okunur."
            return "Yüksek köprü sarkma riski var; açıklık ortası aşağı deforme olur, destek sarkmayı azaltır."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "PA idealin altında; test hızı yükseldikçe köşe şişmesi daha belirgin görünür."
            if result["high_pa_defect"] > 0.33:
                return "PA fazla; yüksek test hızında köşe incelmesi veya küçük boşluk daha görünür olur."
            return "PA ideal aralığa yakın; test hızı yükselse bile köşe davranışı dengeli kalır."
        if mode == "input":
            if params.get("shaper", "MZV") == "Kapalı":
                return "Titreşim sönümleme kapalıyken hız ve ivme arttıkça ringing izleri belirginleşir."
            return "Titreşim sönümleme yüzeydeki ringing izlerini azaltır; agresif shaper ayarları yüzeyi yumuşatabilir."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            if result["restart_gap_risk"] > 0.35:
                return "Geri çekme fazla; ikinci kule başlangıcında yeniden başlama boşluğu (restart gap) görülebilir."
            if result["stringing_risk"] > 0.35:
                return "Geri çekme düşük veya sıcaklık yüksek; boşta hareket sırasında ipliklenme artabilir."
            return "Geri çekme değeri dengeli; boşta hareket sırasında sızıntı kontrollü görünüyor."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] <= 0.75:
                return "Hotend bu hacimsel debiyi rahat karşılıyor; ekstrüzyon hattı dolu görünüyor."
            if result["ratio"] <= 1.0:
                return "Hacimsel debi limite yaklaşıyor; ekstrüzyon hattı hafif incelir ama çoğunlukla süreklidir."
            return "Hotend kapasitesi aşılıyor; ekstrüzyon hattında incelme ve kısa boşluklar oluşur."
        return ""

    @staticmethod
    def recommendation_text(mode, params):
        if mode == "intro":
            return "Soldan bir mod seç, parametreleri değiştir ve 3D sahnedeki temsili sonucu izle."
        if mode == "overhang":
            return "Köprü ortasında sarkma artıyorsa fanı artır, hızı azalt veya destek kullan."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "Köşelerde şişme varsa PA değerini küçük adımlarla artır; yüksek test hızı PA toleransını daraltır."
            if result["high_pa_defect"] > 0.33:
                return "Köşe öncesi boşluk/incelme varsa PA değerini azalt; test hızını düşürmek hatayı yumuşatır."
            return "PA aralığı iyi görünüyor; test hızı arttıkça tolerans daraldığı için hızlı testlerle de doğrula."
        if mode == "input":
            return "Ringing belirginse ivmeyi azalt veya MZV/EI/2HUMP_EI shaper tiplerini karşılaştır."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            if result["restart_gap_risk"] > 0.35:
                return "Yeniden başlama boşluğu (restart gap) varsa geri çekme mesafesini azalt."
            if result["stringing_risk"] > 0.35:
                return "İpliklenme varsa geri çekme mesafesini artır, sıcaklığı düşür veya boşta hareket hızını artır."
            return "Geri çekme dengeli görünüyor."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            line_ratio = result["line_to_nozzle_ratio"]
            if line_ratio < FLOW_REASONABLE_LINE_RATIO_MIN:
                return "Çizgi genişliği nozzle çapına göre düşük; yüzey kaplama zayıf temsil edilebilir."
            if line_ratio > FLOW_REASONABLE_LINE_RATIO_MAX:
                return "Çizgi genişliği nozzle çapına göre yüksek; akış yükü ve kalite riski artabilir."
            return "Çizgi genişliği/nozzle oranı makul; debi yüksekse hız, katman yüksekliği veya çizgi genişliği azaltılabilir."
        return ""

    @staticmethod
    def formula_text(mode, params):
        if mode == "intro":
            return "Temsil notu: 3D sahne eğitim amaçlı ölçeklendirilmiştir; gerçek fizik motoru kullanılmaz."
        if mode == "overhang":
            return "Sarkma = maksimum sarkma × risk × konum etkisi; risk açıklık, fan, hız ve destek ile hesaplanır."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            return f"İdeal PA: {result['ideal']:.3f}, efektif tolerans: ±{result['effective_tolerance']:.3f}, test hızı etki çarpanı: {result['defect_amplifier']:.2f}x."
        if mode == "input":
            return "Dalga = amplitude × sin(2πx / wavelength) × exp(-decay × x)."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            return f"İdeal geri çekme: {result['ideal']:.1f} mm; güvenli aralık {result['safe_min']:.1f}-{result['safe_max']:.1f} mm."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            return f"Hacimsel debi = Katman yüksekliği × Çizgi genişliği × Baskı hızı = {result['flow']:.2f} mm³/s; nozzle çapı formüle doğrudan girmez."
        return ""

    @staticmethod
    def calculated_value_text(mode, params):
        if mode == "overhang":
            span_mm = bridge_scene_config(params)["span_mm"]
            return f"Köprü açıklığı: {span_mm:.0f} mm | Sarkma riski: {FDMModel.overhang_risk(params) * 100:.0f}%"
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            return f"PA: {result['pa']:.3f} | Test hızı: {result['speed']:.0f} mm/s | Tolerans: ±{result['effective_tolerance']:.3f} | Etki çarpanı: {result['defect_amplifier']:.2f}x"
        if mode == "input":
            return f"Titreşim izi/ringing riski: {FDMModel.input_shaping_risk(params) * 100:.0f}%"
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            return f"İpliklenme {result['stringing_risk'] * 100:.0f}% | Yeniden başlama boşluğu {result['restart_gap_risk'] * 100:.0f}%"
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            return f"Hacimsel debi: {result['flow']:.2f} mm³/s | Limit: {float(params.get('max_flow', 12)):.1f} mm³/s | Nozzle çapı: {result['nozzle_diameter']:.1f} mm | Çizgi genişliği/nozzle oranı: {result['line_to_nozzle_ratio']:.2f}x"
        return "3D öğretici görünüm hazır"

    @staticmethod
    def visual_note_text(mode):
        if mode == "intro":
            return "Görsel etki eğitim amaçlı ve temsili gösterilmiştir."
        return "Görsel etki eğitim amacıyla büyütülmüş/temsili gösterilmiştir."

    @staticmethod
    def flow_nozzle_note(params):
        result = FDMModel.volumetric_flow_risk(params)
        line_ratio = result["line_to_nozzle_ratio"]
        if line_ratio < FLOW_REASONABLE_LINE_RATIO_MIN:
            warning = "Çizgi genişliği nozzle'a göre düşük."
        elif line_ratio > FLOW_REASONABLE_LINE_RATIO_MAX:
            warning = "Çizgi genişliği nozzle'a göre yüksek; kalite/akış riski artabilir."
        else:
            warning = "Çizgi genişliği seçilen nozzle için makul aralıkta."
        return (
            "Nozzle çapı debi formülüne doğrudan girmez; çizgi genişliği/nozzle oranı, "
            f"görsel ölçek ve kalite yorumu için kullanılır. Uyarı: {warning}"
        )

    @staticmethod
    def what_happened_text(mode, params):
        if mode == "intro":
            return "Bir FDM baskı parametresi eğitim sahnesi hazır."
        if mode == "overhang":
            risk = FDMModel.overhang_risk(params)
            if risk < 0.20:
                return "Köprüleme temiz kaldı."
            if risk < 0.60:
                return "Köprü ortasında hafif sarkma oluştu."
            return "Köprü ortasında belirgin sarkma oluştu."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "Köşelerde fazla filament birikimi oluştu."
            if result["high_pa_defect"] > 0.33:
                return "Köşe yakınında incelme veya küçük boşluk oluştu."
            return "Köşe davranışı dengeli görünüyor."
        if mode == "input":
            risk = FDMModel.input_shaping_risk(params)
            if risk > 0.35:
                return "Duvar kenarında titreşim izi/ringing oluştu."
            return "Titreşim izi/ringing düşük seviyede kaldı."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            if result["restart_gap_risk"] > 0.35:
                return "Başlangıç noktasında yeniden başlama boşluğu (restart gap) oluştu."
            if result["stringing_risk"] > 0.35:
                return "Boşta hareket sırasında ipliklenme oluştu."
            return "Boşta hareket ve yeniden başlama davranışı dengeli görünüyor."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] > 1.0:
                return "Ekstrüzyon hatlarında incelme ve kısa boşluklar oluştu."
            if result["ratio"] > 0.75:
                return "Ekstrüzyon hattı limite yaklaşırken hafif inceldi."
            return "Ekstrüzyon hatları dolu ve sürekli kaldı."
        return ""

    @staticmethod
    def why_happened_text(mode, params):
        if mode == "intro":
            return "Sahneler parametre etkilerini temsili olarak karşılaştırmak için hazırlanmıştır."
        if mode == "overhang":
            return "Köprü açıklığı, fan, hız ve destek durumuna göre soğuma/sarkma riski değişti."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "PA değeri idealin altında ve test hızı hatayı görünür hale getiriyor."
            if result["high_pa_defect"] > 0.33:
                return "PA değeri idealin üstünde ve test hızı incelmeyi görünür hale getiriyor."
            return "PA değeri ideal aralığa yakın olduğu için basınç dengeli kaldı."
        if mode == "input":
            risk = FDMModel.input_shaping_risk(params)
            if risk <= 0.35:
                return "Shaper ve hız/ivme ayarları titreşimi düşük tuttu."
            return "Yüksek hız/ivme mekanik titreşimi görünür hale getiriyor."
        if mode == "retraction":
            return "Geri çekme, sıcaklık ve boşta hareket hızı sızıntı ya da yeniden başlama riskini belirliyor."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] <= 0.75:
                return "Hesaplanan debi hotend limitinin altında kaldı."
            if result["ratio"] <= 1.0:
                return "Hesaplanan debi hotend limitine yaklaşıyor."
            return "Hesaplanan debi hotend limitini aşıyor."
        return ""

    @staticmethod
    def what_to_do_text(mode, params):
        if mode == "intro":
            return "Soldan bir mod seçip parametreleri değiştir."
        if mode == "overhang":
            return "Fanı artır, hızı azalt veya destek kullan."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "PA değerini küçük adımlarla artır."
            if result["high_pa_defect"] > 0.33:
                return "PA değerini küçük adımlarla azalt."
            return "PA değerini koru; farklı test hızlarında doğrula."
        if mode == "input":
            risk = FDMModel.input_shaping_risk(params)
            if risk <= 0.35:
                return "Ayarları koru; daha yüksek hızlarda shaper tiplerini karşılaştır."
            return "İvmeyi azalt veya shaper tipini karşılaştır."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            if result["restart_gap_risk"] > 0.35:
                return "Geri çekme mesafesini azalt."
            if result["stringing_risk"] > 0.35:
                return "Geri çekme mesafesini artır, sıcaklığı düşür veya boşta hareket hızını artır."
            return "Ayarları koru; malzemeye göre küçük testlerle doğrula."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] <= 0.75:
                return "Ayarlar güvenli görünüyor; çizgi genişliği/nozzle oranını da kontrol et."
            return "Baskı hızını, katman yüksekliğini veya çizgi genişliğini azalt."
        return ""

    @staticmethod
    def progress_items(mode, params):
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            return [("Köşe kalite skoru", result["quality"], False)]
        if mode == "overhang":
            return [("Köprü sarkma riski", FDMModel.overhang_risk(params), True)]
        if mode == "input":
            return [("Titreşim izi/ringing riski", FDMModel.input_shaping_risk(params), True)]
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            return [
                ("İpliklenme riski", result["stringing_risk"], True),
                ("Yeniden başlama boşluğu riski", result["restart_gap_risk"], True),
                ("Genel risk", result["combined_risk"], True),
            ]
        if mode == "flow":
            return [("Hotend limit riski", FDMModel.volumetric_flow_risk(params)["risk"], True)]
        return []

    @staticmethod
    def report_copy_text(mode, params, term_mode=DEFAULT_TERM_MODE):
        # kopyalanan rapor da seçili terim modunu takip etsin
        score_label, _, score, is_risk = FDMModel.score_for_mode(mode, params)
        lines = [
            "FDM Parametreleri Görselleştiricisi",
            f"Mod: {mode_label(mode, term_mode)}",
        ]
        if mode == "intro":
            lines.append("Durum: Simülasyon hazır")
        else:
            lines.append(FDMModel.score_display_text(mode, score_label, score, is_risk))
            lines.append(f"Hesaplanan değer: {FDMModel.calculated_value_text(mode, params)}")
        lines.extend(["", "Parametreler:"])
        if params:
            lines.extend(FDMModel.formatted_parameter_lines(mode, params))
        else:
            lines.append("- Ana sayfa: parametre yok")
        lines.extend(
            [
                "",
                f"Açıklama: {FDMModel.explanation_text(mode, params)}",
                f"Ne oldu: {FDMModel.what_happened_text(mode, params)}",
                f"Neden: {FDMModel.why_happened_text(mode, params)}",
                f"Ne yapmalı: {FDMModel.what_to_do_text(mode, params)}",
                f"Tavsiye: {FDMModel.recommendation_text(mode, params)}",
                f"Formül/Metod: {FDMModel.formula_text(mode, params)}",
            ]
        )
        return "\n".join(lines)


class SimulationState:
    def __init__(self):
        # ekranda seçili olan temel şeyleri burada tutuyoruz
        self.active_mode = "intro"
        self.running = True
        self.animation_time = 0.0
        self.animation_speed = 1.0
        self.term_mode = DEFAULT_TERM_MODE
        self.selected_preset = "PLA"
        self.parameters = deepcopy(DEFAULT_PARAMETERS)
        self.apply_preset("PLA")

    def reset(self):
        self.animation_time = 0.0

    def set_mode(self, mode):
        if mode in MODE_LABELS:
            # mod değişince animasyon başa dönsün, ayarlar olduğu gibi kalsın
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
            # özel seçimi sadece etiket, değerleri ezmeye gerek yok
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
        self.view.setBackgroundColor("#091018")
        layout.addWidget(self.view)
        self.scene_badge = QLabel("Eğitimsel 3D görsel", self)
        self.scene_badge.setObjectName("SceneBadge")
        self.scene_badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.scene_badge.adjustSize()
        self.scene_badge.raise_()
        self.update_mode_scene(reset_camera=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "scene_badge"):
            self.scene_badge.adjustSize()
            self.scene_badge.move(max(12, self.width() - self.scene_badge.width() - 14), 12)
            self.scene_badge.raise_()

    def setup_camera(self, mode=None):
        if self.view is None:
            return
        mode = mode or self.state.active_mode
        presets = {
            "intro": ((0, 0, 4.0), 155.0, 28, -42),
            "overhang": ((0, 0, 3.2), 154.0, 20, -72),
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
        grid = gl.GLGridItem(color=(90, 106, 118, 28))
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
        # model tarafı küçük sayılarla rahat, sahnede okunması için büyütüyoruz
        point = np.asarray(point, dtype=float)
        return np.asarray((point[0] * PART_SCALE, point[1] * PART_SCALE, point[2] * Z_VISUAL_SCALE), dtype=float)

    def scene_width(self, width):
        return max(0.65, float(width) * PART_SCALE * LINE_WIDTH_VISUAL)

    def scene_height(self, height):
        return max(0.22, float(height) * Z_VISUAL_SCALE * LAYER_HEIGHT_VISUAL)

    def mesh_item(self, vertexes, faces, color="#ff8a3d", alpha=1.0, edge="#101820", gl_options=None, draw_edges=True, face_colors=None, smooth=False, compute_normals=False):
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
            smooth=smooth,
            drawEdges=draw_edges,
            edgeColor=color_tuple(edge, min(0.72, 0.12 + alpha * 0.55)),
            computeNormals=compute_normals,
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
        # bağlı filament parçalarını tek mesh gibi çizmek daha temiz duruyor
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
        # random yok, aynı ayarda aynı ufak pürüzler görünsün
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
        if segment.meta.get("profile") == "pressure" and segment.meta.get("pa_corners"):
            point = np.asarray(sample["point"], dtype=float)
            corners = [np.asarray(corner, dtype=float) for corner in segment.meta.get("pa_corners", [])]
            if corners:
                distance = min(float(np.linalg.norm(point[:2] - corner[:2])) for corner in corners)
                radius = max(float(segment.meta.get("pa_corner_radius", zone)), 0.001)
                return smoothstep(1.0 - distance / radius)
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
            line_to_nozzle_ratio = float(meta.get("line_to_nozzle_ratio", 1.0))
            severity = clamp((ratio - 0.75) / 0.75, 0.0, 1.0)
            high_line_load = smoothstep(clamp((line_to_nozzle_ratio - FLOW_REASONABLE_LINE_RATIO_MAX) / 0.55, 0.0, 1.0))
            seed = int(meta.get("line_index", 0)) * 37 + int(meta.get("layer_index", 0)) * 19
            noise = self.deterministic_noise(index, seed)
            noise_b = self.deterministic_noise(index * 3 + 17, seed + 53)
            slow_pulse = 0.5 + 0.5 * math.sin(2 * math.pi * (sample["visible_progress"] * (1.7 + noise_b * 2.2) + noise * 0.41))
            thinning = severity * (0.07 + 0.20 * noise + 0.16 * slow_pulse) + risk * 0.05 + high_line_load * 0.10
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
        # köşe, flow ve retraction izleri aynı kalınlık listesine işleniyor
        modifiers = np.ones(len(samples), dtype=float)
        modifiers = self.apply_corner_blob_profile(modifiers, samples)
        modifiers = self.apply_underextrusion_profile(modifiers, samples)
        modifiers = self.apply_restart_profile(modifiers, samples, total)
        return np.clip(modifiers, 0.18, 1.95)

    def apply_gap_profile(self, samples, total):
        # boşlukları ayrı obje değil, mesh üzerinde görünmeyen aralık gibi tutuyoruz
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
            if high > 0.78 and influence > 0.78 and (near_start or near_end) and corner_key not in pressure_gap_added:
                gaps.append(self.sample_gap_range(samples, index, total, 0.46 + high * 0.18))
                pressure_gap_added.add(corner_key)

            ratio = float(meta.get("flow_ratio", 0.0))
            if ratio > 1.0:
                risk = clamp(float(meta.get("flow_risk", segment.defect_strength)), 0.0, 1.0)
                line_to_nozzle_ratio = float(meta.get("line_to_nozzle_ratio", 1.0))
                high_line_load = smoothstep(clamp((line_to_nozzle_ratio - FLOW_REASONABLE_LINE_RATIO_MAX) / 0.55, 0.0, 1.0))
                seed = int(meta.get("line_index", 0)) * 41 + int(meta.get("layer_index", 0)) * 23
                noise = self.deterministic_noise(index * 5 + 11, seed)
                local_cluster = self.deterministic_noise(int(sample["visible_progress"] * 31), seed + 71)
                gap_probability = clamp((ratio - 1.0) * 0.095 + risk * 0.050 + high_line_load * 0.035, 0.0, 0.22)
                away_from_ends = 0.035 < sample["visible_progress"] < 0.965
                if away_from_ends and noise < gap_probability * (0.55 + local_cluster):
                    gaps.append(self.sample_gap_range(samples, index, total, 0.36 + risk * 0.30 + high_line_load * 0.12))

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
            if meta.get("bridge_profile", False):
                left_anchor = float(meta.get("bridge_anchor_left", sample["point"][0]))
                right_anchor = float(meta.get("bridge_anchor_right", left_anchor + 0.001))
                anchor_min = min(left_anchor, right_anchor)
                anchor_max = max(left_anchor, right_anchor)
                layer_factor = clamp(float(meta.get("overhang_layer_factor", 1.0)), 0.0, 1.0)
                span_progress = clamp((sample["point"][0] - anchor_min) / max(anchor_max - anchor_min, 0.001), 0.0, 1.0)
                mid_span_factor = max(0.0, math.sin(math.pi * span_progress))
                sag_strength = clamp(float(meta.get("bridge_sag_strength", risk)), 0.0, 1.0)
                max_sag = 0.085 + 0.265 * layer_factor
                offsets[index] -= sag_strength * (mid_span_factor ** 1.40) * max_sag
                continue
            root_x = float(meta.get("overhang_root_x", sample["point"][0]))
            tip_x = float(meta.get("overhang_tip_x", root_x + 0.001))
            layer_factor = clamp(float(meta.get("overhang_layer_factor", 1.0)), 0.0, 1.0)
            angle_factor = clamp(float(meta.get("overhang_angle_factor", risk)), 0.0, 1.0)
            cooling_factor = clamp(float(meta.get("overhang_cooling_factor", 0.0)), 0.0, 1.0)
            speed_factor = clamp(float(meta.get("overhang_speed_factor", 0.0)), 0.0, 1.0)
            support_factor = clamp(float(meta.get("overhang_support_factor", 1.0)), 0.0, 1.0)
            free_end_factor = clamp((sample["point"][0] - root_x) / max(tip_x - root_x, 0.001), 0.0, 1.0)
            sag_driver = clamp(0.72 * angle_factor + 0.18 * cooling_factor + 0.10 * speed_factor, 0.0, 1.0)
            max_sag = 0.030 + 0.125 * layer_factor
            offsets[index] -= max(risk, sag_driver * 0.35 * support_factor) * (free_end_factor ** 1.55) * max_sag
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
        # filament hattı kutu gibi değil, biraz yuvarlak ve yumuşak görünsün
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
        section_count = BEAD_CROSS_SECTION_SEGMENTS
        section_angles = [-math.pi / 2 + 2 * math.pi * side / section_count for side in range(section_count)]
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
            for angle in section_angles:
                lateral = perp * (half * math.cos(angle))
                vertical = z * (height * (0.04 + 0.44 * math.sin(angle)))
                vertices.append(point + lateral + vertical)

        faces = []
        face_colors = []
        side_dark = self.shaded_color(color, alpha, 82)
        side_mid = self.shaded_color(color, alpha, 96)
        top_light = self.shaded_color(color, alpha, 114)
        palette = []
        for angle in section_angles:
            if math.sin(angle) > 0.45:
                palette.append(top_light)
            elif math.sin(angle) < -0.55:
                palette.append(side_dark)
            else:
                palette.append(side_mid)

        def add_tri(face, rgba):
            faces.append(face)
            face_colors.append(rgba)

        def add_quad(a, b, c, d, rgba):
            add_tri((a, b, d), rgba)
            add_tri((b, c, d), rgba)

        def add_cap(section, reverse=False):
            base = section * section_count
            rgba = side_dark
            for offset in range(1, section_count - 1):
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

            first = index * section_count
            second = (index + 1) * section_count
            for side in range(section_count):
                next_side = (side + 1) % section_count
                add_quad(first + side, first + next_side, second + next_side, second + side, palette[side])

            if not next_visible:
                add_cap(index + 1, reverse=False)

        if not faces:
            return None
        return self.add_item(self.mesh_item(vertices, faces, color, alpha, draw_edges=False, face_colors=face_colors, smooth=True, compute_normals=True))

    def add_box(self, center, size, color="#ff8a3d", alpha=1.0, edge="#101820", draw_edges=True):
        cx, cy, cz = center
        sx, sy, sz = size
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2
        vertexes = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        return self.add_item(self.mesh_item(vertexes, faces, color, alpha, edge, draw_edges=draw_edges))

    def add_cylinder(self, center, radius, height, color="#ff8a3d", alpha=1.0, segments=ROUND_GEOMETRY_SEGMENTS):
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
        return self.add_item(self.mesh_item(vertexes, faces, color, alpha, edge="#0f1419", draw_edges=False, smooth=True, compute_normals=True))

    def add_cone(self, tip, radius, height, color="#cbd5df", alpha=1.0, segments=ROUND_GEOMETRY_SEGMENTS):
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
        return self.add_item(self.mesh_item(vertexes, faces, color, alpha, edge="#687482", draw_edges=False, smooth=True, compute_normals=True))

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

    def draw_support_column(self, segment, progress=1.0):
        partial = segment.partial(progress) if progress < 0.999 else segment
        start = self.scene_point(partial.start)
        end = self.scene_point(partial.end)
        z0 = min(start[2], end[2])
        z1 = max(start[2], end[2])
        if z1 - z0 < 0.05:
            return
        size_x = max(0.95, float(segment.meta.get("support_size_x", 0.16)) * PART_SCALE)
        size_y = max(3.2, float(segment.meta.get("support_size_y", 0.92)) * PART_SCALE)
        center = (start[0], start[1], (z0 + z1) * 0.5)
        self.add_box(center=center, size=(size_x, size_y, z1 - z0), color=segment.color, alpha=0.26, edge="#438eb8")

    def draw_bead_run(self, run):
        if not run:
            return
        has_flow_marks = any(segment.segment_type == "underextrusion" or float(segment.meta.get("flow_ratio", 0.0)) > 0.75 for segment in run)
        has_pressure_marks = any(segment.meta.get("profile") == "pressure" for segment in run)
        has_restart_marks = any(float(segment.meta.get("restart_gap_risk", 0.0)) > 0.0 or float(segment.meta.get("restart_blob_risk", 0.0)) > 0.0 for segment in run)
        max_step = 0.10 if (has_flow_marks or has_pressure_marks or has_restart_marks) else 0.12
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
        if risk < 0.15:
            return
        if risk < 0.30:
            count = 1
        else:
            count = max(3, int(round(risk * 6)))
        count = min(6, count)
        start = np.asarray(segment.start, dtype=float)
        end = np.asarray(segment.end, dtype=float)
        layer_seed = int(segment.meta.get("layer_index", 0))
        for index in range(count):
            centered = index - (count - 1) / 2
            y_offset = centered * lerp(0.018, 0.055, risk)
            start_jitter = (self.deterministic_noise(index, layer_seed) - 0.5) * 0.025
            end_jitter = (self.deterministic_noise(index + 11, layer_seed) - 0.5) * 0.025
            sag = lerp(0.018, 0.085, risk) * (0.75 + 0.16 * abs(centered))
            curve = [
                start + np.asarray((0.00, y_offset + start_jitter, -0.18), dtype=float),
                (start + end) * 0.5 + np.asarray((0.0, y_offset * 0.38, -0.22 - sag), dtype=float),
                end + np.asarray((0.00, y_offset * 0.45 + end_jitter, -0.18), dtype=float),
            ]
            partial = partial_polyline3d(curve, progress)
            if len(partial) >= 2:
                alpha = 0.24 if risk < 0.30 else lerp(0.35, 0.55, risk)
                self.add_line([self.scene_point(point) for point in partial], "#ffbf7a", 0.50, alpha)

    def draw_segment(self, segment, progress=1.0, active=False):
        if segment.segment_type == "support_column":
            self.draw_support_column(segment, progress if active else 1.0)
            return
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

    def toolhead_scale(self):
        ns = NOZZLE_SCALE * 1.26
        if self.state.active_mode == "flow":
            flow_result = FDMModel.volumetric_flow_risk(self.state.current_params())
            nozzle_diameter = float(flow_result["nozzle_diameter"])
            ns *= lerp(0.88, 1.28, clamp((nozzle_diameter - 0.4) / 0.4, 0.0, 1.0))
        return ns

    def add_nozzle(self, tip):
        tip = self.scene_point(tip) + np.array((0.0, 0.0, self.scene_height(self.engine.visual_segment_height()) * 0.72))
        ns = self.toolhead_scale()
        x, y, z = tip

        # cad gibi kasmayalım, uzaktan okunan sade toolhead yeter
        self.add_box(center=(x, y, z + 13.4 * ns), size=(9.2 * ns, 6.3 * ns, 3.6 * ns), color="#46525d", alpha=1.0, edge="#8fa0ad", draw_edges=False)
        self.add_box(center=(x, y, z + 10.7 * ns), size=(6.9 * ns, 4.9 * ns, 1.0 * ns), color="#6d7a84", alpha=1.0, edge="#b7c3cc", draw_edges=False)
        for fin_index in range(4):
            fin_z = z + (8.0 + fin_index * 0.78) * ns
            self.add_box(center=(x, y, fin_z), size=(6.2 * ns, 4.6 * ns, 0.28 * ns), color="#a6b0b8", alpha=1.0, edge="#d4dce2", draw_edges=False)
        self.add_box(center=(x, y, z + 5.85 * ns), size=(7.4 * ns, 3.7 * ns, 2.25 * ns), color="#c0a06f", alpha=1.0, edge="#ffe0a8", draw_edges=False)
        self.add_box(center=(x + 5.0 * ns, y - 0.15 * ns, z + 8.6 * ns), size=(2.3 * ns, 4.0 * ns, 3.0 * ns), color="#26313b", alpha=1.0, edge="#61717f", draw_edges=False)
        self.add_box(center=(x + 6.35 * ns, y - 0.15 * ns, z + 6.25 * ns), size=(2.2 * ns, 2.0 * ns, 1.05 * ns), color="#26313b", alpha=1.0, edge="#5b6b78", draw_edges=False)
        self.add_cone(tip, radius=2.25 * ns, height=5.15 * ns, color="#d7b56f", alpha=1.0)
        self.add_cylinder(center=(x, y, z + 0.34 * ns), radius=0.50 * ns, height=0.72 * ns, color="#ff8a3d", alpha=0.95)
        self.add_line([(x, y, z + 15.2 * ns), (x, y, z + 21.0 * ns)], "#c9d4dc", 3.2, 0.78)
        self.add_line([(x + 2.9 * ns, y + 1.4 * ns, z + 14.5 * ns), (x + 6.5 * ns, y + 2.0 * ns, z + 18.8 * ns)], "#657482", 2.3, 0.58)

    def add_cooling_airflow(self):
        if self.state.active_mode != "overhang":
            return
        params = self.state.current_params()
        fan = clamp(float(params.get("fan", 0)) / 100.0, 0.0, 1.0)
        if fan <= 0.01:
            return

        # hava akışı sahneye değil, nozzle yanındaki fan bloğuna bağlı dursun
        ns = self.toolhead_scale()
        nozzle_tip = self.scene_point(self.engine.nozzle_position) + np.array((0.0, 0.0, self.scene_height(self.engine.visual_segment_height()) * 0.72))
        origin_base = nozzle_tip + np.array((7.1 * ns, -1.6 * ns, 6.2 * ns))
        airflow_direction = np.array((-0.46, -0.06, -0.89), dtype=float)
        airflow_direction /= float(np.linalg.norm(airflow_direction))
        side_axis = np.array((0.02, 1.0, -0.08), dtype=float)
        side_axis /= float(np.linalg.norm(side_axis))
        lift_axis = np.array((0.22, 0.02, 0.97), dtype=float)
        line_count = max(1, 1 + int(round(fan * 5)))
        alpha = lerp(0.12, 0.34, fan)
        width = lerp(0.55, 1.15, fan)
        length = lerp(3.2, 6.9, fan) * ns
        drift_span = lerp(0.6, 1.6, fan) * ns

        for index in range(line_count):
            centered = index - (line_count - 1) / 2
            lane_spacing = lerp(0.34, 0.72, fan) * ns
            phase = self.state.animation_time * 0.95 + index * 0.31
            drift = (phase % 1.0) * drift_span
            lane_offset = side_axis * centered * lane_spacing
            flutter = lift_axis * math.sin(phase * math.tau) * 0.10 * ns * fan
            start = origin_base + lane_offset + flutter + airflow_direction * drift
            end = start + airflow_direction * length
            mid = start + airflow_direction * (length * 0.52) + side_axis * math.sin(phase * math.tau + 0.6) * 0.16 * ns * fan
            self.add_line([start, mid, end], "#7dd3fc", width, alpha)

    def add_overhang_supports(self):
        if self.state.active_mode != "overhang":
            return
        params = self.state.current_params()
        if not bool(params.get("support", False)):
            return
        config = bridge_scene_config(params)
        z_bottom = self.engine.visual_layer_z(0) - 0.03
        z_top = self.engine.visual_layer_z(config["pillar_layers"]) - 0.105
        if z_top <= z_bottom:
            return
        for support_x in config["support_x_positions"]:
            center = self.scene_point((float(support_x), 0.0, (z_bottom + z_top) * 0.5))
            size = (0.14 * PART_SCALE, 0.88 * PART_SCALE, (z_top - z_bottom) * Z_VISUAL_SCALE)
            self.add_box(center=center, size=size, color="#7dd3fc", alpha=0.24, edge="#438eb8")

    def build_static_scene(self):
        self.add_grid()
        self.add_print_bed()
        self.add_overhang_supports()

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
        # her karede sahneyi baştan kuruyoruz, obje birikmesi olmasın
        self.engine.update_progress(self.state.animation_time)
        self.clear_scene()
        self.build_static_scene()
        self.draw_completed_segments()
        self.draw_active_segment_progress()
        self.update_nozzle_pose()
        self.add_cooling_airflow()

    def animate(self):
        self.update_scene()


class ParameterPanel(QWidget):
    parameters_changed = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setObjectName("ParameterPanel")
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
        # sağ panel komple yenileniyor, çünkü her modun kontrol takımı farklı
        self.clear_layout()
        title = QLabel(mode_label(self.state.active_mode, self.state.term_mode))
        title.setObjectName("ModeTitle")
        title.setWordWrap(True)
        self.layout.addWidget(title)
        if self.state.active_mode == "intro":
            self.build_intro_panel()
        elif self.state.active_mode == "overhang":
            self.add_help("İki destek arasındaki köprü çizgilerinde fan, hız ve açıklığın sarkmayı nasıl etkilediğini gösterir.")
            self.add_section_title("Parametreler")
            self.build_overhang_controls()
        elif self.state.active_mode == "pressure":
            self.add_help("Basınç dengeleme (Pressure Advance), köşe ve hız değişimlerinde filament basıncını dengelemeyi temsil eder.")
            self.add_section_title("Parametreler")
            self.build_pressure_controls()
        elif self.state.active_mode == "input":
            self.add_help("Shaper tipi, hız ve ivme değerlerinin yüzeydeki titreşim izi/ringing etkisini gösterir.")
            self.add_section_title("Parametreler")
            self.build_input_controls()
        elif self.state.active_mode == "retraction":
            self.add_help("Boşta hareket sırasında oluşan ipliklenme ve fazla geri çekme kaynaklı yeniden başlama boşluğu (restart gap) riskini gösterir.")
            self.add_section_title("Parametreler")
            self.build_retraction_controls()
        elif self.state.active_mode == "flow":
            self.add_help("Katman yüksekliği, çizgi genişliği ve baskı hızından hotend'in taşıması gereken hacimsel debiyi hesaplar.")
            self.add_section_title("Parametreler")
            self.build_flow_controls()
        self.layout.addStretch(1)

    def build_intro_panel(self):
        self.add_help("Soldan bir mod seç; ayarların 3D sahnede nasıl temsil edildiğini izle.")

    def add_help(self, text):
        body = QLabel(text)
        body.setWordWrap(True)
        body.setObjectName("ModeDescription")
        body.setMaximumHeight(36)
        self.layout.addWidget(body)

    def add_section_title(self, text):
        label = QLabel(text)
        label.setObjectName("SectionLabel")
        self.layout.addWidget(label)

    def current_value(self, key, default):
        return self.state.current_params().get(key, default)

    def add_slider(self, label_text, key, min_value, max_value, step, unit, tooltip):
        # slider değerini direkt state'e yazar, sahne de oradan okur
        frame = QFrame()
        frame.setObjectName("ControlRow")
        frame.setMinimumHeight(50)
        layout = QGridLayout(frame)
        layout.setContentsMargins(9, 5, 9, 5)
        layout.setVerticalSpacing(4)
        title = QLabel(label_text)
        title.setObjectName("ControlTitle")
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

        decimals = 0 if step >= 1 else 1 if step >= 0.1 else 2 if step >= 0.01 else 3

        def format_value(value):
            if decimals == 0:
                return f"{int(round(value))} {unit}".strip()
            return f"{value:.{decimals}f} {unit}".strip()

        def update_value(slider_value):
            value = clamp(min_value + slider_value * step, min_value, max_value)
            value = int(round(value)) if decimals == 0 else round(value, decimals)
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

    def add_combo(self, label_text, key, options, tooltip, unit=""):
        frame = QFrame()
        frame.setObjectName("ControlRow")
        layout = QGridLayout(frame)
        layout.setContentsMargins(9, 5, 9, 5)
        layout.setVerticalSpacing(4)
        title = QLabel(label_text)
        title.setObjectName("ControlTitle")
        value_label = QLabel()
        value_label.setObjectName("ValueLabel")
        value_label.setMinimumWidth(96)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        combo = QComboBox()
        combo.addItems(options)
        combo.setToolTip(tooltip)
        current = str(self.current_value(key, options[0]))
        if current in options:
            combo.setCurrentText(current)

        def format_combo_value(text):
            return f"{text} {unit}".strip()

        def changed(text):
            value_label.setText(format_combo_value(text))
            self.state.update_parameter(key, text)
            self.parameters_changed.emit()

        value_label.setText(format_combo_value(combo.currentText()))
        combo.currentTextChanged.connect(changed)
        layout.addWidget(title, 0, 0)
        layout.addWidget(value_label, 0, 1)
        layout.addWidget(combo, 1, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        self.layout.addWidget(frame)

    def build_overhang_controls(self):
        self.add_slider("Köprü açıklığı", "angle", 10, 80, 1, "mm", "İki destek kulesi arasındaki köprü mesafesini temsil eder.")
        self.add_slider("Fan hızı", "fan", 0, 100, 1, "%", "Köprü filamentinin ne kadar hızlı soğuduğunu temsil eder.")
        self.add_slider("Baskı hızı", "speed", 20, 120, 1, "mm/s", "Köprüleme sırasında yüksek hız sarkmayı artırabilir.")
        self.add_checkbox("Destek kullan", "support", "Boşluğun altında sade destek kolonları gösterir ve sarkma riskini düşürür.")

    def build_pressure_controls(self):
        self.add_slider("Pressure Advance", "pa", PA_SLIDER_MIN, PA_SLIDER_MAX, PA_SLIDER_STEP, "", "Köşe basıncını 0.005 adımlarla dengelemeyi temsil eder.")
        self.add_combo("Ekstruder tipi", "extruder", ["Direct Drive", "Bowden"], "Direct drive ve Bowden için ideal PA farklıdır.")
        self.add_slider("Test hızı", "speed", 30, 180, 1, "mm/s", "Bu hız, PA testinde köşe davranışını zorlaştıran temsili test hızıdır. Yüksek test hızı, düşük/yüksek PA hatalarını daha görünür yapar.")

    def build_input_controls(self):
        self.add_slider("İvme", "acceleration", 500, 10000, 100, "mm/s²", "Yüksek ivme titreşim izi/ringing riskini artırabilir.")
        self.add_slider("Rezonans frekansı", "frequency", 20, 80, 1, "Hz", "Titreşime yatkın frekansı temsil eder.")
        self.add_combo("Shaper tipi", "shaper", ["Kapalı", "MZV", "EI", "2HUMP_EI"], "Input shaping titreşim izlerini azaltır.")
        self.add_slider("Baskı hızı", "speed", 40, 250, 1, "mm/s", "Yüksek hız titreşim etkisini görünür yapabilir.")

    def build_retraction_controls(self):
        self.add_slider("Geri çekme mesafesi", "retraction", 0.0, 8.0, 0.1, "mm", "Boşta hareket sırasında filamentin geri çekilmesini temsil eder.")
        self.add_slider("Nozzle sıcaklığı", "temperature", 180, 260, 1, "°C", "Sıcaklık arttıkça ipliklenme riski artabilir.")
        self.add_slider("Boşta hareket hızı (Travel)", "travel_speed", 50, 250, 1, "mm/s", "Hızlı boşta hareket sızıntı süresini azaltır.")
        self.add_combo("Ekstruder tipi", "extruder", ["Direct Drive", "Bowden"], "İdeal geri çekme mesafesi ekstruder tipine göre değişir.")

    def build_flow_controls(self):
        self.add_slider("Katman yüksekliği", "layer_height", 0.08, 0.40, 0.01, "mm", "Katman yüksekliği debiyi etkiler.")
        self.add_slider("Çizgi genişliği", "line_width", 0.35, 0.80, 0.01, "mm", "Çizgi genişliği debiyi etkiler.")
        self.add_slider("Baskı hızı", "print_speed", 20, 250, 1, "mm/s", "Baskı hızı debiyi doğrudan artırır.")
        self.add_slider("Hotend kapasitesi", "max_flow", 4, 35, 1, "mm³/s", "Hotend'in eritebildiği yaklaşık hacimdir.")
        self.add_combo("Nozzle çapı", "nozzle_diameter", ["0.4", "0.6", "0.8"], "Nozzle çapı tek başına debiyi değiştirmez. Debiyi katman yüksekliği, çizgi genişliği ve baskı hızı belirler; nozzle çapı çizgi genişliğinin mantıklı aralıkta olup olmadığını yorumlamak için kullanılır.", "mm")


class InfoPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setObjectName("InfoPanel")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 8, 10, 8)
        self.layout.setSpacing(5)
        title = QLabel("Sonuç")
        title.setObjectName("CardTitle")
        self.score_label = QLabel()
        self.score_label.setObjectName("ResultScore")
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setTextVisible(False)
        self.calculated = QLabel()
        self.calculated.setObjectName("ResultLine")
        self.calculated.setWordWrap(True)
        self.visual_note = QLabel()
        self.visual_note.setObjectName("TechNote")
        self.visual_note.setWordWrap(True)
        self.what_label = QLabel()
        self.what_label.setObjectName("ResultLine")
        self.what_label.setWordWrap(True)
        self.why_label = QLabel()
        self.why_label.setObjectName("ResultLine")
        self.why_label.setWordWrap(True)
        self.todo_label = QLabel()
        self.todo_label.setObjectName("ResultLine")
        self.todo_label.setWordWrap(True)
        self.layout.addWidget(title)
        self.layout.addWidget(self.score_label)
        self.layout.addWidget(self.score_bar)
        self.layout.addWidget(self.calculated)
        self.layout.addWidget(self.what_label)
        self.layout.addWidget(self.why_label)
        self.layout.addWidget(self.todo_label)
        self.layout.addWidget(self.visual_note)
        self.update_info()

    def compact_technical_note(self, mode, params):
        if mode == "intro":
            return "Görseller eğitim amaçlı temsilidir; gerçek fizik motoru kullanılmaz."
        if mode == "overhang":
            return "Sarkma görseli açıklık, fan, hız ve destek etkisine göre temsil edilir."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            return f"İdeal PA: {result['ideal']:.3f}. Test hızı hataları daha görünür yapar."
        if mode == "input":
            return "Titreşim izi/ringing dalgası temsili olarak ölçeklendirilir."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            return f"Güvenli geri çekme aralığı yaklaşık {result['safe_min']:.1f}-{result['safe_max']:.1f} mm."
        if mode == "flow":
            return "Hacimsel debi = Katman yüksekliği × Çizgi genişliği × Baskı hızı. Nozzle çapı formüle doğrudan girmez."
        return "Görsel etki eğitim amacıyla ölçeklendirilmiştir."

    def observed_text(self, mode, params):
        if mode == "intro":
            return "Bir mod seçildiğinde 3D sahnede temsili etki gösterilir."
        if mode == "overhang":
            risk = FDMModel.overhang_risk(params)
            if risk < 0.20:
                return "Köprü çizgileri stabil görünüyor."
            if risk < 0.60:
                return "Köprünün orta kısmında hafif sarkma görülüyor."
            return "Köprünün orta kısmında belirgin sarkma görülüyor."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "Köşelerde fazla filament birikimi görünüyor."
            if result["high_pa_defect"] > 0.33:
                return "Köşe yakınında incelme veya küçük boşluk görünüyor."
            return "Köşe davranışı dengeli görünüyor."
        if mode == "input":
            if FDMModel.input_shaping_risk(params) > 0.35:
                return "Duvar kenarında titreşim izi/ringing belirginleşiyor."
            return "Titreşim izi/ringing düşük seviyede kalıyor."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            if result["restart_gap_risk"] > 0.35:
                return "Başlangıç noktasında yeniden başlama boşluğu (restart gap) riski öne çıkıyor."
            if result["stringing_risk"] > 0.35:
                return "Boşta hareket sırasında ipliklenme riski artıyor."
            return "Boşta hareket ve yeniden başlama dengeli görünüyor."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] > 1.0:
                return "Çizgilerde incelme ve kısa boşluklar oluşabilir."
            if result["ratio"] > 0.75:
                return "Hacimsel debi limite yaklaşırken çizgi hafif inceliyor."
            return "Ekstrüzyon hatları dolu ve sürekli görünüyor."
        return ""

    def likely_cause_text(self, mode, params):
        if mode == "intro":
            return "Sahne, ayar etkilerini hızlıca karşılaştırmak için sadeleştirilmiştir."
        if mode == "overhang":
            return "Açıklık uzun, fan düşük veya hız yüksek olduğunda filament daha kolay sarkar."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "PA değeri idealin altında kaldığında köşe basıncı geç boşalır."
            if result["high_pa_defect"] > 0.33:
                return "PA değeri fazla olduğunda köşeye yaklaşırken basınç erken düşer."
            return "PA değeri ideal aralığa yakın olduğu için basınç dengeli kalır."
        if mode == "input":
            if FDMModel.input_shaping_risk(params) > 0.35:
                return "Yüksek hız veya ivme mekanik titreşimi daha görünür yapar."
            return "Shaper ve hız/ivme ayarları titreşimi düşük tutuyor."
        if mode == "retraction":
            return "Geri çekme mesafesi, sıcaklık ve boşta hareket hızı birlikte sızıntı dengesini belirler."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] > 1.0:
                return "Gereken debi hotend limitini aşıyor."
            if result["ratio"] > 0.75:
                return "Gereken debi hotend limitine yaklaşıyor."
            return "Gereken debi hotend limitinin altında kalıyor."
        return ""

    def suggestion_text(self, mode, params):
        if mode == "intro":
            return "Bir mod seçip sliderları küçük adımlarla dene."
        if mode == "overhang":
            return "Fanı artırmak, hızı düşürmek veya destek kullanmak sonucu iyileştirebilir."
        if mode == "pressure":
            result = FDMModel.pressure_advance_quality(params)
            if result["low_pa_defect"] > 0.33:
                return "PA değerini küçük adımlarla artırmayı dene."
            if result["high_pa_defect"] > 0.33:
                return "PA değerini küçük adımlarla azaltmayı dene."
            return "Bu PA değerini koruyup farklı test hızlarında doğrula."
        if mode == "input":
            if FDMModel.input_shaping_risk(params) > 0.35:
                return "İvme değerini düşür veya shaper tiplerini karşılaştır."
            return "Ayarları koru; daha yüksek hızlarda shaper tiplerini karşılaştır."
        if mode == "retraction":
            result = FDMModel.retraction_stringing_risk(params)
            if result["restart_gap_risk"] > 0.35:
                return "Geri çekme mesafesini biraz azaltmayı dene."
            if result["stringing_risk"] > 0.35:
                return "Geri çekmeyi artırmak, sıcaklığı düşürmek veya boşta hareket hızını artırmak yardımcı olabilir."
            return "Ayarlar dengeli; malzemeye göre küçük testlerle doğrula."
        if mode == "flow":
            result = FDMModel.volumetric_flow_risk(params)
            if result["ratio"] > 0.75:
                return "Baskı hızını, katman yüksekliğini veya çizgi genişliğini azaltmak debiyi düşürür."
            return "Ayarlar güvenli görünüyor; çizgi genişliği/nozzle oranını da kontrol et."
        return ""

    def update_info(self):
        # alttaki kısa sonuç kartı, uzun rapordan ayrı ve hızlı okunacak yer
        mode = self.state.active_mode
        params = self.state.current_params()
        score_label, normalized, score, is_risk = FDMModel.score_for_mode(mode, params)
        color = QColor("#3ddc84") if mode == "intro" else risk_color(normalized) if is_risk else quality_color(normalized)
        self.score_label.setText(FDMModel.score_display_text(mode, score_label, score, is_risk))
        self.score_label.setStyleSheet(f"color: {color.name()};")
        self.score_bar.setVisible(mode != "intro")
        if mode != "intro":
            self.score_bar.setValue(score)
            self.score_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color.name()}; border-radius: 4px; }}")
        self.calculated.setText(f"Hesaplanan değer: {FDMModel.calculated_value_text(mode, params)}")
        self.what_label.setText(f"Gözlenen durum: {self.observed_text(mode, params)}")
        self.why_label.setText(f"Muhtemel sebep: {self.likely_cause_text(mode, params)}")
        self.todo_label.setText(f"Öneri: {self.suggestion_text(mode, params)}")
        self.visual_note.setText(f"Teknik not: {self.compact_technical_note(mode, params)}")


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
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)
        root_layout.addWidget(self.build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.build_left_panel())
        self.scene_widget = GLSceneWidget(self.state)
        splitter.addWidget(self.scene_widget)
        splitter.addWidget(self.build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([198, 1080, 334])
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self.build_footer())
        self.setCentralWidget(root)

    def build_header(self):
        frame = QFrame()
        frame.setObjectName("Header")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(2)
        title = QLabel("FDM Parametreleri Görselleştiricisi")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Yeni Başlayanlar İçin FDM 3D Baskı Ayarları Görsel Rehberi")
        subtitle.setObjectName("AppSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame

    def build_left_panel(self):
        # sol taraf sabit kalsın, uzun isimleri buton içinde toparlıyoruz
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(198)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.setSpacing(6)
        layout.addWidget(self.panel_title("Modlar"))
        self.mode_buttons = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for mode in MODE_KEYS:
            button = QPushButton(mode_button_label(mode, self.state.term_mode))
            button.setObjectName("ModeButton")
            button.setCheckable(True)
            button.setFixedHeight(48)
            button.setToolTip(mode_label(mode, self.state.term_mode))
            button.clicked.connect(lambda checked=False, selected_mode=mode: self.set_mode(selected_mode))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            layout.addWidget(button)
        self.mode_buttons[self.state.active_mode].setChecked(True)
        layout.addStretch(1)
        return panel

    def build_right_panel(self):
        panel = QFrame()
        panel.setObjectName("RightPanel")
        panel.setMinimumWidth(318)
        panel.setMaximumWidth(386)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.setSpacing(7)
        self.parameter_panel = ParameterPanel(self.state)
        self.parameter_panel.parameters_changed.connect(self.parameters_changed)
        scroll = QScrollArea()
        scroll.setObjectName("ParameterScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setObjectName("ParameterViewport")
        scroll.viewport().setAutoFillBackground(False)
        scroll.setWidget(self.parameter_panel)
        layout.addWidget(scroll, 1)
        self.info_panel = InfoPanel(self.state)
        layout.addWidget(self.info_panel, 0)
        return panel

    def build_footer(self):
        footer = QFrame()
        footer.setObjectName("Footer")
        footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(7)
        for text, slot in [
            ("Başlat", self.start_animation),
            ("Durdur", self.stop_animation),
            ("Sıfırla", self.reset_animation),
        ]:
            button = QPushButton(text)
            button.setObjectName("ToolbarButton")
            button.setFixedHeight(30)
            button.clicked.connect(slot)
            toolbar.addWidget(button)

        toolbar.addSpacing(8)
        toolbar.addWidget(QLabel("Animasyon hızı"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(20)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(186)
        self.speed_slider.valueChanged.connect(self.set_animation_speed_from_slider)
        toolbar.addWidget(self.speed_slider)
        self.speed_value = QLabel("1.00x")
        self.speed_value.setObjectName("ValueLabel")
        self.speed_value.setFixedWidth(54)
        toolbar.addWidget(self.speed_value)

        toolbar.addSpacing(8)
        toolbar.addWidget(QLabel("Ön Ayar"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([preset_display_name(name) for name in ["PLA", "PETG", "ABS", "Custom"]])
        self.preset_combo.setCurrentText(preset_display_name(self.state.selected_preset))
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        self.preset_combo.setFixedWidth(96)
        toolbar.addWidget(self.preset_combo)

        toolbar.addSpacing(8)
        toolbar.addWidget(QLabel("Terim Modu"))
        self.term_mode_combo = QComboBox()
        self.term_mode_combo.addItems(list(TERM_MODE_LABELS.keys()))
        self.term_mode_combo.setCurrentText(self.state.term_mode)
        self.term_mode_combo.currentTextChanged.connect(self.set_term_mode)
        self.term_mode_combo.setFixedWidth(118)
        toolbar.addWidget(self.term_mode_combo)

        export_button = QPushButton("PNG Kaydet")
        export_button.setObjectName("SecondaryToolbarButton")
        export_button.setFixedHeight(30)
        export_button.clicked.connect(self.export_png)
        toolbar.addWidget(export_button)
        self.copy_button = QPushButton("Raporu Kopyala")
        self.copy_button.setObjectName("SecondaryToolbarButton")
        self.copy_button.setFixedHeight(30)
        self.copy_button.clicked.connect(self.copy_report)
        toolbar.addWidget(self.copy_button)
        toolbar.addStretch(1)

        self.footer_explanation = QLabel()
        self.footer_explanation.setObjectName("StatusText")
        self.footer_explanation.setWordWrap(False)
        self.footer_explanation.setMaximumHeight(18)
        layout.addLayout(toolbar)
        layout.addWidget(self.footer_explanation)
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

    def refresh_mode_labels(self):
        for mode, button in self.mode_buttons.items():
            button.setText(mode_button_label(mode, self.state.term_mode))
            button.setToolTip(mode_label(mode, self.state.term_mode))

    def set_term_mode(self, term_mode):
        if term_mode not in TERM_MODE_LABELS or term_mode == self.state.term_mode:
            return
        # sadece yazılar değişsin, aktif mod ve animasyonla oynamıyoruz
        self.state.term_mode = term_mode
        self.refresh_mode_labels()
        self.parameter_panel.rebuild()
        self.info_panel.update_info()
        self.update_footer()

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
        # combo ekranda özel der, state tarafında custom diye gezer
        preset_name = preset_internal_name(preset_name)
        self._updating_preset = True
        self.state.apply_preset(preset_name)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(preset_display_name(preset_name))
        self.preset_combo.blockSignals(False)
        self.state.reset()
        self.refresh_panels(rebuild_parameters=True)
        self._updating_preset = False

    def parameters_changed(self):
        if not self._updating_preset and self.state.selected_preset != "Custom":
            # elle oynandıysa artık hazır preset değil
            self.state.selected_preset = "Custom"
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(preset_display_name("Custom"))
            self.preset_combo.blockSignals(False)
        self.state.reset()
        self.refresh_panels(rebuild_parameters=False)

    def refresh_panels(self, rebuild_parameters=False):
        # parametre paneli pahalı değil ama gerekmiyorsa yeniden kurmuyoruz
        if rebuild_parameters:
            self.parameter_panel.rebuild()
        self.info_panel.update_info()
        self.update_footer()
        self.scene_widget.update_mode_scene(reset_camera=rebuild_parameters)

    def update_footer(self):
        mode_messages = {
            "intro": "Soldan bir mod seç; 3D sahnede temsili etkiyi izle.",
            "overhang": "Köprü açıklığı, fan, hız ve destek sarkma görünümünü etkiler.",
            "pressure": "Test hızı, PA köşe hatalarının görünürlüğünü artırır.",
            "input": "Shaper ve ivme ayarı titreşim izi/ringing görünümünü değiştirir.",
            "retraction": "Geri çekme ayarı ipliklenme ve yeniden başlama boşluğu dengesini gösterir.",
            "flow": "Hacimsel debi hesabı katman yüksekliği, çizgi genişliği ve baskı hızı ile yapılır.",
        }
        self.footer_explanation.setText(
            "Parametre etkileri eğitim amacıyla temsili olarak gösterilir; gerçek fizik motoru kullanılmaz. "
            f"{mode_messages.get(self.state.active_mode, '')}"
        )

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
        text = FDMModel.report_copy_text(self.state.active_mode, self.state.current_params(), self.state.term_mode)
        QApplication.clipboard().setText(text)
        old_text = self.copy_button.text()
        self.copy_button.setText("Kopyalandı")
        QTimer.singleShot(1200, lambda: self.copy_button.setText(old_text))

    def apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #0a0f15; color: #edf3f8; }
            QWidget { color: #edf3f8; font-family: Segoe UI, Arial; font-size: 10pt; }
            QMenuBar { background: #101720; color: #dce6ee; border-bottom: 1px solid #1c2935; }
            QMenuBar::item:selected, QMenu::item:selected { background: #243342; }
            QMenu { background: #111a23; color: #edf3f8; border: 1px solid #273746; }
            #Header, #SidePanel, #RightPanel, #Footer, #InfoPanel, #GLSceneShell {
                background: #111922; border: 1px solid #223140; border-radius: 8px;
            }
            #Header { background: #121d27; }
            #AppTitle { font-size: 15pt; font-weight: 700; color: #ffffff; }
            #AppSubtitle { color: #a4b4c3; font-size: 9pt; }
            #PanelTitle { font-size: 10.5pt; font-weight: 700; color: #ffffff; }
            #ModeTitle { font-size: 13pt; font-weight: 800; color: #ffffff; }
            #ModeDescription { color: #b3c0cc; font-size: 9.2pt; line-height: 125%; }
            #SectionLabel { color: #f0a76b; font-size: 8.8pt; font-weight: 800; padding-top: 2px; }
            #ControlTitle { color: #eaf0f6; font-weight: 600; }
            #CardTitle { font-weight: 700; color: #f3f7fb; }
            #MutedText { color: #aebaca; }
            #BodyText { color: #dce6ee; }
            #WarningText { color: #ffcf7a; font-weight: 600; }
            #StatusText { color: #9fb0c0; font-size: 8.7pt; }
            #ResultScore { font-weight: 800; font-size: 11pt; }
            #ResultLine { color: #d5e3ef; font-size: 9pt; line-height: 125%; }
            #TechNote {
                color: #8ea4b5; font-size: 8.6pt; padding-top: 3px;
                border-top: 1px solid #243444;
            }
            #SceneBadge {
                color: #a9bbc8; background: rgba(13, 20, 28, 150);
                border: 1px solid #263746; border-radius: 7px;
                padding: 3px 7px; font-size: 8pt; font-weight: 600;
            }
            #InfoCard, #HelpBox, #ControlRow, #MetricBox {
                background: #141f2a; border: 1px solid #253545; border-radius: 7px;
            }
            #MetricBox, #FormulaText {
                padding: 7px; color: #cfe8ff; background: #0f161f; border: 1px solid #253546; border-radius: 8px;
            }
            #ErrorBox {
                color: #ffb3b8; background: #27151a; border: 1px solid #63303b; border-radius: 8px; padding: 14px;
            }
            QPushButton {
                background: #1b2835; color: #edf3f8; border: 1px solid #304355; border-radius: 7px; padding: 5px 8px;
                min-height: 22px;
            }
            QPushButton:hover { background: #243545; border-color: #486477; }
            QPushButton:pressed { background: #17222d; }
            QPushButton#ModeButton { text-align: left; padding: 6px 8px; line-height: 120%; }
            QPushButton#ModeButton:checked {
                background: #263849; border-color: #ff8a3d; color: #ffffff;
                font-weight: 700;
            }
            QPushButton#ToolbarButton { min-width: 76px; }
            QPushButton#SecondaryToolbarButton {
                min-width: 92px; background: #162230; color: #cbd8e3; border-color: #2a3c4c;
            }
            QPushButton#SecondaryToolbarButton:hover { background: #1e2c3a; color: #edf3f8; border-color: #3b5366; }
            QComboBox {
                background: #0f161f; color: #edf3f8; border: 1px solid #2d4153; border-radius: 6px; padding: 5px 8px;
                min-height: 22px;
            }
            QComboBox QAbstractItemView { background: #111922; color: #edf3f8; selection-background-color: #25384a; }
            QSlider::groove:horizontal { height: 6px; background: #2a3a48; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #ff8a3d; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 16px; height: 16px; margin: -6px 0; background: #f5f8fb;
                border: 2px solid #ff8a3d; border-radius: 8px;
            }
            QCheckBox {
                background: #141f2a; border: 1px solid #253545; border-radius: 7px; padding: 6px;
            }
            QProgressBar {
                background: #2a3a48; border: 0; border-radius: 4px; height: 8px;
            }
            QSplitter::handle { background: #090d13; }
            #ParameterPanel, #ParameterViewport { background: transparent; }
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { background: #0c1219; width: 10px; margin: 2px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #314456; min-height: 36px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            #Separator { color: #263645; background: #263645; }
            #ValueLabel { color: #f5b17a; font-weight: 700; }
            """
        )


def configure_open_gl_surface():
    surface_format = QSurfaceFormat()
    surface_format.setSamples(8)
    surface_format.setDepthBufferSize(24)
    surface_format.setStencilBufferSize(8)
    surface_format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(surface_format)
    if pg is not None:
        pg.setConfigOptions(antialias=True)


def main():
    configure_open_gl_surface()
    app = QApplication(sys.argv)
    app.setApplicationName("FDM Parametreleri Görselleştiricisi")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
