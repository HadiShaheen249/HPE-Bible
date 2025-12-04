"""
Main entry point for RTMPose + ByteTrack Pose Estimator
"""

import sys
from pathlib import Path
import argparse

# Import classes
from pose_estimator import RTMPoseEstimator
from config import Config
from app import PoseEstimatorApp


def quick_test():
    """Quick system test"""
    print("\n🧪 Running quick test...")
    
    try:
        # Print paths
        Config.print_paths()
        Config.print_model_info()
        
        # Create Estimator object
        estimator = RTMPoseEstimator()
        print("✅ System is ready!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_cli():
    """Run from command line"""
    parser = argparse.ArgumentParser(
        description='RTMPose + ByteTrack Pose Estimation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py --image path/to/image.jpg
  python main.py --video path/to/video.mp4
  python main.py --camera
  python main.py --det-conf 0.7 --pose-conf 0.6 --image test.jpg
  python main.py --output custom/path/result.jpg --image test.jpg
        """
    )
    
    # Arguments
    parser.add_argument('--image', '-i', type=str, help='Image path')
    parser.add_argument('--video', '-v', type=str, help='Video path')
    parser.add_argument('--camera', '-c', action='store_true', help='Use camera')
    
    parser.add_argument('--det-conf', type=float, default=None,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--pose-conf', type=float, default=None,
                       help='Pose confidence threshold (0.0-1.0)')
    
    parser.add_argument('--output', '-o', type=str, help='Output path (optional)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--no-show', action='store_true', help='Do not show results')
    parser.add_argument('--no-track', action='store_true', help='Disable tracking')
    
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda:0 or cpu)')
    
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--show-paths', action='store_true', help='Show all paths and exit')
    parser.add_argument('--test', action='store_true', help='Run system test')
    
    args = parser.parse_args()
    
    # Show paths and exit
    if args.show_paths:
        Config.print_paths()
        Config.print_model_info()
        return
    
    # Run test
    if args.test:
        quick_test()
        return
    
    # Create Estimator object
    estimator = RTMPoseEstimator(
        det_conf=args.det_conf,
        pose_conf=args.pose_conf,
        device=args.device
    )
    
    # Disable tracking if requested
    if args.no_track:
        estimator.tracker = None
        print("⚠️  Tracking disabled")
    
    # Process image
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        
        estimator.predict_image(
            image_path=args.image,
            save_result=not args.no_save,
            output_path=args.output
        )
    
    # Process video
    elif args.video:
        if not Path(args.video).exists():
            print(f"❌ Video not found: {args.video}")
            sys.exit(1)
        
        estimator.predict_video(
            video_path=args.video,
            save_result=not args.no_save,
            output_path=args.output,
            show_live=not args.no_show
        )
    
    # Use camera
    elif args.camera:
        estimator.predict_video(
            video_path=0,
            save_result=not args.no_save,
            output_path=args.output,
            show_live=True
        )
    
    # Interactive mode
    elif args.interactive:
        app = PoseEstimatorApp()
        app.run_interactive()
    
    else:
        parser.print_help()


def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║      🤸‍♂️ RTMPose + ByteTrack Pose Estimation 🤸‍♀️          ║
║                                                           ║
║     Professional Pose Estimation with Tracking            ║
║          Powered by OpenMMLab & ByteTrack                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)
    Config.print_paths()


def show_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print("📋 Choose operation mode:")
    print("="*60)
    print("1. Interactive Mode")
    print("2. Quick Image Processing")
    print("3. Quick Video Processing")
    print("4. Run Camera")
    print("5. Test System")
    print("6. Show Help")
    print("7. Show Paths & Model Info")
    print("0. Exit")
    print("="*60)


def main():
    """Main function"""
    # Print banner
    print_banner()
    
    # If command line arguments exist
    if len(sys.argv) > 1:
        run_cli()
        return
    
    # Show main menu
    while True:
        show_menu()
        choice = input("\n👉 Your choice: ").strip()
        
        if choice == '1':
            # Interactive mode
            app = PoseEstimatorApp()
            app.run_interactive()
        
        elif choice == '2':
            # Process image
            path = input("📁 Enter image path: ").strip()
            if Path(path).exists():
                estimator = RTMPoseEstimator()
                estimator.predict_image(path)
            else:
                print("❌ File not found!")
        
        elif choice == '3':
            # Process video
            path = input("📁 Enter video path: ").strip()
            if Path(path).exists():
                estimator = RTMPoseEstimator()
                estimator.predict_video(path)
            else:
                print("❌ File not found!")
        
        elif choice == '4':
            # Camera
            print("📹 Opening camera... (Press Q to exit)")
            estimator = RTMPoseEstimator()
            estimator.predict_video(0)
        
        elif choice == '5':
            # Test system
            quick_test()
        
        elif choice == '6':
            # Help
            print_help()
        
        elif choice == '7':
            # Show paths and model info
            Config.print_paths()
            Config.print_model_info()
        
        elif choice == '0':
            print("\n👋 Thank you for using the application!")
            break
        
        else:
            print("❌ Invalid choice!")
        
        input("\n⏸️  Press Enter to continue...")


def print_help():
    """Print help documentation"""
    help_text = """
╔═══════════════════════════════════════════════════════════╗
║                    📖 User Guide                          ║
╚═══════════════════════════════════════════════════════════╝

🔹 Command Line Usage:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Process image (auto-saves to output/images/)
python main.py --image path/to/image.jpg

# Process video (auto-saves to output/videos/)
python main.py --video path/to/video.mp4

# Use camera (auto-saves with timestamp)
python main.py --camera

# Set confidence thresholds
python main.py --det-conf 0.7 --pose-conf 0.6 --video test.mp4

# Custom output path
python main.py --image test.jpg --output custom/path/result.jpg

# Do not save results
python main.py --video test.mp4 --no-save

# Do not show results
python main.py --video test.mp4 --no-show

# Disable tracking (pose only)
python main.py --video test.mp4 --no-track

# Use CPU instead of GPU
python main.py --device cpu --image test.jpg

# Show all paths
python main.py --show-paths

# Test system
python main.py --test

# Interactive mode
python main.py --interactive


🔹 Available Models:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detection (RTMDet):
  - rtmdet-m  → Medium (Default)

Pose Estimation (RTMPose):
  - rtmpose-t → Tiny (Fast)
  - rtmpose-s → Small
  - rtmpose-m → Medium (Default) ✓
  - rtmpose-l → Large

Tracking:
  - ByteTrack v2 (Always enabled)

ℹ️  Models are automatically downloaded on first use


🔹 Output Locations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Images:  output/images/output_[filename]
Videos:  output/videos/output_[filename]
Camera:  output/videos/camera_output_[timestamp].mp4
JSON:    output/json/
CSV:     output/csv/


🔹 Keyboard Controls:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q  → Exit video processing


🔹 Directory Structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

project/
├── models/
│   ├── detection/      ← RTMDet models
│   └── pose/          ← RTMPose models
├── configs/
│   ├── detection/      ← Detection configs
│   └── pose/          ← Pose configs
├── input/
│   ├── images/        ← Place input images here
│   └── videos/        ← Place input videos here
└── output/
    ├── images/        ← Processed images
    ├── videos/        ← Processed videos
    ├── json/          ← JSON results
    └── csv/           ← CSV results


🔹 Installation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Install dependencies
pip install -r requirements.txt

# Install OpenMMLab packages using mim
pip install openmim
mim install mmdet mmpose


🔹 Practical Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Simple run without parameters
python main.py

# Process with high confidence
python main.py --det-conf 0.8 --pose-conf 0.7 --video dance.mp4

# Quick camera recording with tracking
python main.py --camera

# Process without saving (preview only)
python main.py --video test.mp4 --no-save

# Process on CPU
python main.py --device cpu --image photo.jpg


🔹 Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Multi-person pose estimation
✓ Real-time person tracking (ByteTrack)
✓ 17 COCO keypoints detection
✓ GPU acceleration support
✓ Video and image processing
✓ Live camera support
✓ Tracking ID persistence
✓ High accuracy with RTMPose


🔹 For More Help:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python main.py --help

╚═══════════════════════════════════════════════════════════╝
    """
    print(help_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)