# Skellicap

Give me a video of an athlete running from a lateral perspective and I give you a dataset of the position of key skeletal points on each frame

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

### Using Makefile

To process a video file:
```bash
make process INPUT_VIDEO=path/to/video.mp4
```
You can also specify a custom output path:
```bash
make process INPUT_VIDEO=path/to/video.mp4 OUTPUT_JSON=my_results.json
```

To analize a trackpoint file:
```bash
make analize OUTPUT_JSON=path/to/trackpoints.json ANALYZED_JSON=analized_results.json
```

### As a Module

```python
import cv2
from skellicap import PoseTracker, PoseAnalyzer

# Initialize tracker
tracker = PoseTracker()
frame = cv2.imread("running_human.jpg")
landmarks = tracker.process_frame(frame)

if landmarks:
    # Analyze the results
    analyzer = PoseAnalyzer()
    # Note: analyze_results expects a list of frame data as produced by process_video
    # or a specifically formatted dictionary for single frames.
    analysis = analyzer.analyze_results([{"frame_index": 0, "landmarks": landmarks}])
    
    print(f"Knee Angle: {analysis[0]['analysis']['left']['knee_angle']}")
```

### From Command Line

```bash
# Process video
python -m skellicap.pose_tracker --input path/to/video.mp4 --output results.json

# Analyze results
python -m skellicap.pose_analyzer --input results.json --output analyzed.json
```

## Detected Landmarks

For each frame, the following skeletal points are detected for both left and right sides:
- Hip
- Knee
- Ankle

