import json
import math
import numpy as np

class PoseAnalyzer:
    def __init__(self, y_vel_threshold=2.0, y_pos_threshold=15.0, min_stride_frames=10):
        """
        Args:
            y_vel_threshold: Max vertical velocity to consider foot stationary (pixels/frame).
            y_pos_threshold: Max distance from the lowest recorded point to consider foot on ground (pixels).
            min_stride_frames: Minimum frames between consecutive strides for a single foot.
        """
        self.y_vel_threshold = y_vel_threshold
        self.y_pos_threshold = y_pos_threshold
        self.min_stride_frames = min_stride_frames

    def calculate_angle(self, p1, p2, p3):
        """Calculates the angle at p2 given points p1, p2, p3."""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)

    def analyze_results(self, tracker_output):
        # Handle both new {metadata, frames} and old [...] formats
        if isinstance(tracker_output, dict):
            tracker_results = tracker_output.get('frames', [])
            metadata = tracker_output.get('metadata', {})
        else:
            tracker_results = tracker_output
            metadata = {}

        if not tracker_results:
            return {}

        fps = metadata.get('fps', 30.0) # Default to 30 if missing
        ms_per_frame = 1000.0 / fps
        resolution_error_ms = ms_per_frame  # Error bound is +/- 1 frame duration

        # Find global max Y (lowest point in image) for ankles to establish ground level
        left_ankle_ys = [f['landmarks']['left']['ankle']['y'] for f in tracker_results if f['landmarks']]
        right_ankle_ys = [f['landmarks']['right']['ankle']['y'] for f in tracker_results if f['landmarks']]
        
        max_y_left = max(left_ankle_ys) if left_ankle_ys else 0
        max_y_right = max(right_ankle_ys) if right_ankle_ys else 0

        analyzed_frames = []
        prev_landmarks = None

        for frame_data in tracker_results:
            frame_idx = frame_data['frame_index']
            landmarks = frame_data['landmarks']
            
            if not landmarks:
                analyzed_frames.append(frame_data)
                continue

            frame_analysis = {
                'frame_index': frame_idx,
                'landmarks': landmarks,
                'analysis': {
                    'left': {},
                    'right': {}
                }
            }

            for side in ['left', 'right']:
                side_data = landmarks[side]
                hip = side_data['hip']
                knee = side_data['knee']
                ankle = side_data['ankle']

                # 1. Calculate Knee Angle
                knee_angle = self.calculate_angle(hip, knee, ankle)
                frame_analysis['analysis'][side]['knee_angle'] = knee_angle

                # 2. Calculate Velocities
                vx, vy, v_agg = 0.0, 0.0, 0.0
                if prev_landmarks and prev_landmarks[side]:
                    prev_ankle = prev_landmarks[side]['ankle']
                    vx = ankle['x'] - prev_ankle['x']
                    vy = ankle['y'] - prev_ankle['y']
                    v_agg = math.sqrt(vx**2 + vy**2)

                frame_analysis['analysis'][side]['velocity'] = {
                    'x': vx,
                    'y': vy,
                    'aggregate': v_agg
                }

                # 3. Ground Contact Detection
                # Higher Y in pixel coordinates is lower in the frame (closer to ground)
                ground_ref = max_y_left if side == 'left' else max_y_right
                is_near_ground = (ground_ref - ankle['y']) <= self.y_pos_threshold
                is_stationary_y = abs(vy) <= self.y_vel_threshold
                
                frame_analysis['analysis'][side]['is_ground_contact'] = bool(is_near_ground and is_stationary_y)

            analyzed_frames.append(frame_analysis)
            prev_landmarks = landmarks

        # Detect Strides
        strides = self.detect_strides(analyzed_frames)
        
        # Post-process strides for temporal metrics
        for s in strides:
            s['duration_ms'] = s['duration'] * ms_per_frame
            s['resolution_error_ms'] = resolution_error_ms

        return {
            'metadata': {
                **metadata,
                'ms_per_frame': ms_per_frame,
                'resolution_error_ms': resolution_error_ms
            },
            'frames': analyzed_frames,
            'gait_metrics': {
                'strides': strides
            }
        }

    def detect_strides(self, analyzed_frames):
        """
        Detects strides based on the transition to ground contact.
        """
        strides = []
        last_global_contact = None  # {side, frame_index, x}
        last_side_contact_frame = {'left': -self.min_stride_frames, 'right': -self.min_stride_frames}
        in_contact = {'left': False, 'right': False}

        for frame in analyzed_frames:
            if not frame.get('analysis'):
                continue
            
            current_frame = frame['frame_index']
            for side in ['left', 'right']:
                is_currently_on_ground = frame['analysis'][side]['is_ground_contact']
                
                # Detect Initial Contact (False -> True)
                if is_currently_on_ground and not in_contact[side]:
                    # Check threshold
                    if (current_frame - last_side_contact_frame[side]) >= self.min_stride_frames:
                        current_x = frame['landmarks'][side]['ankle']['x']
                        
                        stride_entry = {
                            'side': side,
                            'frame_initiated': current_frame,
                            'duration': 0,
                            'x_distance': 0.0
                        }
                        
                        if last_global_contact:
                            stride_entry['duration'] = current_frame - last_global_contact['frame_index']
                            stride_entry['x_distance'] = abs(current_x - last_global_contact['x'])
                        
                        strides.append(stride_entry)
                        
                        # Update last contact info
                        last_global_contact = {
                            'side': side,
                            'frame_index': current_frame,
                            'x': current_x
                        }
                        last_side_contact_frame[side] = current_frame
                
                in_contact[side] = is_currently_on_ground
        
        return strides

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Analyze Pose Tracking Data")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file from PoseTracker")
    parser.add_argument("--output", type=str, default="analyzed_results.json", help="Path to save analyzed results")
    parser.add_argument("--y-vel", type=float, default=2.0, help="Y velocity threshold for ground contact")
    parser.add_argument("--y-pos", type=float, default=15.0, help="Y position threshold for ground contact")
    parser.add_argument("--min-stride-frames", type=int, default=10, help="Minimum frames between strides for a foot")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        exit(1)

    with open(args.input, "r") as f:
        data = json.load(f)

    analyzer = PoseAnalyzer(
        y_vel_threshold=args.y_vel, 
        y_pos_threshold=args.y_pos,
        min_stride_frames=args.min_stride_frames
    )
    results = analyzer.analyze_results(data)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Analyzed results saved to {args.output}")
