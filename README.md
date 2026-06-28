# Skellicap

A Python package for human pose tracking and kinematic analysis of running mechanics using MediaPipe.

## Installation

### Using Makefile (Recommended)

```bash
make setup
```

### Manual Installation

1. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Workflow

1. **Process Video**: Extract raw landmarks.
   ```bash
   make process INPUT_VIDEO=path/to/video.mp4
   ```

2. **Analyze Results**: Calculate angles, velocities, and ground contact.
   ```bash
   make analyze
   ```

3. **Visualize**: Open `dashboard.html` in your browser and upload the generated `analyzed_results.json`.

### Dashboard Features
- **Interactive Plots**: Hover over charts to see frame-by-frame data.
- **Topline Metrics**: Min/max knee angles and average ground contact time.
- **Quartile Analysis**: Performance summary table divided into four segments of the run.

### Programmatic Usage

```python
import cv2
from skellicap import PoseTracker, PoseAnalyzer

tracker = PoseTracker()
frame = cv2.imread("running.jpg")
landmarks = tracker.process_frame(frame)

if landmarks:
    analyzer = PoseAnalyzer()
    analysis = analyzer.analyze_results([{"frame_index": 0, "landmarks": landmarks}])
    print(f"Left Knee Angle: {analysis[0]['analysis']['left']['knee_angle']} degrees")
```

## Detected Landmarks

For each frame, the following skeletal points are detected for both left and right sides:
- Hip
- Knee
- Ankle
