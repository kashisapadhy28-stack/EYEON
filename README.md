# EYEON

## Intelligent Object & Person Tracking System

EYEON is an AI-based computer vision system designed to detect and track people and objects in a monitored environment, associate objects with people, analyze their position and zones, and generate explainable alerts for a security operator.

The system is designed as an operator-assistance tool and does not autonomously take harmful or irreversible actions.

---

## Features

### Core Features

* Person and object detection using YOLO
* Multi-object tracking with unique IDs
* Person-object association
* Temporal ownership estimation
* Zone-based position tracking
* Movement analysis
* Explainable risk scoring
* Controlled alert generation
* Event logging
* Alert snapshots
* Monitoring dashboard

### Initial Object Classes

The first version focuses on common objects:

* Person
* Backpack
* Handbag
* Suitcase
* Cell phone
* Laptop
* Bottle

Custom object classes may be introduced in a later development phase.

---

## Architecture

```text
Camera / Video
      ↓
Object Detection
      ↓
Multi-Object Tracking
      ↓
Person-Object Association
      ↓
Position & Zone Analysis
      ↓
Movement Analysis
      ↓
Risk Engine
      ↓
Alert Generation
      ↓
Event Logging
      ↓
Dashboard
```

A visual architecture diagram will be added later.

---

## Project Structure

```text
EYEON/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── association/
│   ├── positioning/
│   ├── risk/
│   ├── alerts/
│   ├── database/
│   └── utils/
│
├── dashboard/
│
├── data/
│   ├── videos/
│   ├── images/
│   └── annotations/
│
├── models/
│
├── evaluation/
│
├── notebooks/
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Technology Stack

* Python 3.11
* Ultralytics YOLO
* OpenCV
* NumPy
* Pandas
* Shapely
* PyYAML
* SQLite
* Streamlit

---

## Installation

The project uses a Python virtual environment.

Check the Python version:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Verify setup and dependencies:

```bash
python verify_setup.py
```

---

## Configuration

Project settings are maintained in:

```text
config.yaml
```

The configuration includes:

* Detection model
* Detection confidence thresholds
* Object classes
* Tracking configuration
* Association weights
* Temporal voting
* Zones
* Risk weights
* Alert cooldown
* Database location
* Dashboard settings

This allows system parameters to be changed without modifying the core source code. Settings can be accessed across modules using `from src.utils.config_loader import load_config, get_path`.

---

## Development Roadmap

### Phase 0 — Project Setup

* [x] Project structure (19 subdirectories with `.gitkeep`)
* [x] Python package modularization (`__init__.py` in all packages)
* [x] Python virtual environment (`.venv`)
* [x] Dependencies installed (`ultralytics`, `opencv-python`, `numpy`, `pandas`, `shapely`, `pyyaml`, `streamlit`, `scipy`, `lapx`)
* [x] Central configuration file (`config.yaml`)
* [x] Dynamic config loader (`src/utils/config_loader.py`)
* [x] Pretrained model weights (`models/yolo11n.pt`)
* [x] Initial test videos (`data/videos/`)
* [x] Git ignore configuration (`.gitignore`)
* [x] Environment verification script (`verify_setup.py`)
* [x] Comprehensive documentation (`README.md`)

### Phase 1 — Detection & Tracking

* [x] Video input (`src/detection/video_input.py`, `VideoStream` generator, HUD & FPS tracking)
* [x] YOLO detection (`src/detection/detector.py`, `ObjectDetector`, target classes & per-category confidence filtering)
* [x] Person tracking (`src/tracking/tracker.py`, persistent IDs, movement history trails)
* [x] Object tracking (ByteTrack & BoT-SORT support for everyday monitored objects)
* [x] Tracking quality testing (`evaluation/test_tracker.py`, `evaluation/compare_trackers.py`)

### Phase 2 — Person-Object Intelligence

* [x] Person-object association (`src/association/association.py`, spatial IoU, distance, body region, motion vector consistency)
* [x] Temporal ownership voting (15-frame rolling window, 70% confidence threshold, stranger interference rejection)
* [x] Object state management (`ObjectState`: `CARRIED`, `DETACHED`, `UNATTENDED`, 5.0s timeout alert trigger)

### Phase 3 — Position & Movement

* [x] Person foot-point positioning (`src/positioning/zones.py`)
* [x] Zone detection (`ZoneManager`, polygon membership test via `cv2.pointPolygonTest`)
* [x] Dwell time tracking and transition event emission (`ENTERED`, `INSIDE`, `EXITED`)
* [x] Perspective transformation (`src/positioning/perspective.py`, Planar Homography)
* [x] Movement analysis (`src/positioning/movement.py`, velocity, direction & loitering detection)

### Phase 4 — Risk & Alerts

* [x] Explainable risk engine (`src/risk/risk_engine.py`, 4-factor composite risk score)
* [x] Alert state machine (`src/alerts/alert_manager.py`, `AlertState`: `ACTIVE`, `ACKNOWLEDGED`, `RESOLVED`)
* [x] Event logging (`src/database/db.py`, SQLite persistence schema)
* [x] Alert snapshots (`src/alerts/alert_manager.py`, frame snapshot capture on alert trigger)

### Phase 5 — Dashboard & Evaluation

* [x] Monitoring dashboard (`dashboard/app.py`, Streamlit-based security operator UI)
* [x] Event history (`dashboard/app.py`, interactive security incident register)
* [x] Alert snapshots gallery (`dashboard/app.py`, high-risk snapshot viewer)
* [x] Analytics & Risk Metrics (`dashboard/app.py`, distribution stats & metrics)
* [x] System benchmarking script (`evaluation/benchmark.py`, FPS, frame latency, object/track counts)

### Phase 6 — Advanced Development

* [ ] Custom object detection model
* [ ] Multi-camera support
* [ ] Model optimization
* [ ] Safe hardware integration

---

## Evaluation

The system will be evaluated using:

| Metric                       | Status          |
| ---------------------------- | --------------- |
| Detection Precision / Recall | To be evaluated |
| Tracking IDF1                | To be evaluated |
| MOTA                         | To be evaluated |
| ID Switches                  | To be evaluated |
| Association Accuracy         | To be evaluated |
| Alert Precision / Recall     | To be evaluated |
| False Alerts / Hour          | To be evaluated |
| FPS                          | To be evaluated |
| End-to-End Latency           | To be evaluated |

No performance values will be reported until they are measured experimentally.

---

## Limitations

* Detection may produce false positives or false negatives.
* Crowded scenes may cause tracking ID switches.
* Small or partially occluded objects can be difficult to detect.
* Person-object association may become uncertain during occlusion or object handover.
* Risk scores are explainable estimates and should not be treated as definitive judgments.
* Human operator review remains part of the alert workflow.

---

## Privacy

EYEON is designed with privacy-aware operation in mind.

Planned measures include:

* Avoiding face recognition unless explicitly required.
* Limiting stored event data.
* Configurable data retention.
* Storing relevant event snapshots rather than unnecessary continuous recordings.
* Using the system as an operator-assistance tool.

---

## Current Status

**Phases 0 through 5 — Fully Completed**

The end-to-end EYEON system (Detection -> Tracking -> Association -> Positioning & Movement -> Risk Engine -> Alert State Machine -> Database Persistence -> Streamlit Dashboard & System Benchmark) is complete, operational, and verified with unit tests.

---

## Future Scope

Future development may include:

* Custom object detection models
* Improved re-identification
* Multi-camera tracking
* Real-world coordinate mapping
* GPU optimization
* Real-time deployment
* Safe PTZ camera integration
* Operator notification systems

---

## License

To be decided.

---

## Author

**Kashisa Padhy**

GitHub: `kashisapadhy28-stack`
