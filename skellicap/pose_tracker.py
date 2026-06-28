import cv2
import mediapipe as mp
import numpy as np

class PoseTracker:
    def __init__(self, static_image_mode=False, model_complexity=1, 
                 smooth_landmarks=True, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarks_indices = {
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }

    def process_frame(self, frame):
        """
        Processes a single frame and returns the positions of hip, knee, and ankle.
        
        Args:
            frame: BGR image (OpenCV format).
            
        Returns:
            dict: Dictionary containing landmarks for left and right sides.
                  Returns None if no pose is detected.
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        h, w, _ = frame.shape
        landmarks = {
            'left': {},
            'right': {}
        }
        
        mapping = {
            'left_hip': ('left', 'hip'),
            'right_hip': ('right', 'hip'),
            'left_knee': ('left', 'knee'),
            'right_knee': ('right', 'knee'),
            'left_ankle': ('left', 'ankle'),
            'right_ankle': ('right', 'ankle')
        }

        for name, index in self.landmarks_indices.items():
            landmark = results.pose_landmarks.landmark[index]
            side, part = mapping[name]
            landmarks[side][part] = {
                'x': int(landmark.x * w),
                'y': int(landmark.y * h),
                'z': landmark.z,
                'visibility': landmark.visibility
            }
            
        return landmarks

    def draw_landmarks(self, frame, landmarks):
        """
        Draws the detected landmarks on the frame.
        """
        if not landmarks:
            return frame

        output_frame = frame.copy()
        
        # Define colors (BGR)
        colors = {
            'left': (0, 255, 0),   # Green
            'right': (0, 0, 255)   # Red
        }

        for side in ['left', 'right']:
            color = colors[side]
            parts = landmarks[side]
            
            # Draw points
            for part_name, coords in parts.items():
                cv2.circle(output_frame, (coords['x'], coords['y']), 5, color, -1)
                cv2.putText(output_frame, f"{side} {part_name}", (coords['x'] + 5, coords['y'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw lines (hip to knee, knee to ankle)
            if 'hip' in parts and 'knee' in parts:
                cv2.line(output_frame, (parts['hip']['x'], parts['hip']['y']),
                         (parts['knee']['x'], parts['knee']['y']), color, 2)
            if 'knee' in parts and 'ankle' in parts:
                cv2.line(output_frame, (parts['knee']['x'], parts['knee']['y']),
                         (parts['ankle']['x'], parts['ankle']['y']), color, 2)

        return output_frame

    def process_video(self, video_path, output_path=None):
        """
        Processes a video file and detects landmarks for each frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.process_frame(frame)
            frame_results.append({
                'frame_index': frame_count,
                'landmarks': landmarks
            })
            frame_count += 1

        cap.release()
        return {
            'metadata': {
                'fps': fps,
                'total_frames': frame_count
            },
            'frames': frame_results
        }

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Pose Tracking on Video/Images")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="results.json", help="Path to save results as JSON")
    args = parser.parse_args()

    tracker = PoseTracker()

    if args.input:
        print(f"Processing video: {args.input}")
        results = tracker.process_video(args.input)
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {args.output}")
    else:
        # Simple test script
        print("PoseTracker module loaded. No input video provided. Running smoke test.")
        # Create a dummy blank image for a basic smoke test
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = tracker.process_frame(dummy_frame)
        print("Smoke test (blank frame) results:", results)
