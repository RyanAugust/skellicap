import argparse
import json
import os
from .pose_tracker import PoseTracker
from .pose_analyzer import PoseAnalyzer

def run_tracker(args):
    tracker = PoseTracker()
    print(f"Processing video: {args.input}")
    results = tracker.process_video(args.input)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output}")

def run_analyzer(args):
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

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

def main():
    parser = argparse.ArgumentParser(description="Skellicap - Human Pose Tracking and Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Track command
    track_parser = subparsers.add_parser("track", help="Extract raw landmarks from video")
    track_parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    track_parser.add_argument("--output", type=str, default="results.json", help="Path to save results as JSON")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Perform kinematic analysis on landmarks")
    analyze_parser.add_argument("--input", type=str, required=True, help="Path to input JSON file from PoseTracker")
    analyze_parser.add_argument("--output", type=str, default="analyzed_results.json", help="Path to save analyzed results")
    analyze_parser.add_argument("--y-vel", type=float, default=2.0, help="Y velocity threshold for ground contact")
    analyze_parser.add_argument("--y-pos", type=float, default=15.0, help="Y position threshold for ground contact")
    analyze_parser.add_argument("--min-stride-frames", type=int, default=10, help="Minimum frames between strides for a foot")

    args = parser.parse_args()

    if args.command == "track":
        run_tracker(args)
    elif args.command == "analyze":
        run_analyzer(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
