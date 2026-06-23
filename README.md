# FDM Simulator

FDM Simulator is an interactive desktop application for exploring common FDM 3D printing behaviors. It helps users experiment with print parameters and see how those choices affect bridging, pressure advance, ringing, stringing, and volumetric flow.

![FDM Simulator main screen](assets/screenshots/main-screen.png)

## Highlights

- Interactive 3D print scene with animated toolpath visualization
- Technical and explanatory terminology modes
- Material presets for PLA, PETG, ABS, and custom settings
- Mode-specific controls for practical print tuning scenarios
- Quality and risk scoring for different print behaviors

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python FDM-simulator.py
```

## Screenshots

![Bridge and cooling simulation](assets/screenshots/bridge-cooling.png)

Bridge and cooling settings show how span length, fan speed, print speed, and support choices affect sagging risk.

![Pressure advance simulation](assets/screenshots/pressure-advance.png)

Pressure Advance mode demonstrates how extrusion compensation changes corner quality.

![Input shaping simulation](assets/screenshots/input-shaping-ringing.png)

Input Shaping mode visualizes ringing artifacts caused by motion and resonance settings.

![Retraction and stringing simulation](assets/screenshots/retraction-stringing.png)

Retraction mode shows how travel movement, temperature, and retraction distance influence stringing.

![Volumetric flow simulation](assets/screenshots/volumetric-flow.png)

Volumetric Flow mode highlights the relationship between layer height, line width, print speed, and hotend capacity.
