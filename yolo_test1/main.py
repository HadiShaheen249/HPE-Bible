"""
Main entry point for YOLOv8 Pose Estimator
Main entry point
"""

import sys
from pathlib import Path
import argparse

# Import classes
from pose_estimator import YOLOv8PoseEstimator
from config import Config
from app import PoseEstimatorApp


def quick_test():
    """Quick system test"""
    print("\n🧪 Running quick test...")
    
    try:
        # Print paths
        Config.print_paths()
        
        # Create Estimator object
        estimator = YOLOv8PoseEstimator()
        print("✅ System is ready!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_cli():
    """Run from command line"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Pose Estimation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py --image path/to/image.jpg
  python main.py --video path/to/video.mp4
  python main.py --camera
  python main.py --model yolov8m-pose.pt --conf 0.7 --image test.jpg
        """
    )
    
    # Arguments
    parser.add_argument('--image', '-i', type=str, help='Image path')
    parser.add_argument('--video', '-v', type=str, help='Video path')
    parser.add_argument('--camera', '-c', action='store_true', help='Use camera')
    parser.add_argument('--model', '-m', type=str, default='yolov8n-pose.pt', 
                       help='Model type (n/s/m/l/x)')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--output', '-o', type=str, help='Output path (optional)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--no-show', action='store_true', help='Do not show results')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--show-paths', action='store_true', help='Show all paths and exit')
    
    args = parser.parse_args()
    
    # Show paths and exit
    if args.show_paths:
        Config.print_paths()
        return
    
    # Create Estimator
    estimator = YOLOv8PoseEstimator(
        model_name=args.model,
        conf_threshold=args.conf
    )
    
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
    ║          🤸‍♂️  YOLOv8 Pose Estimation Tool  🤸‍♀️             ║
    ║                                                           ║
    ║           Professional Pose Estimation Tool               ║
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
    print("7. Show Paths")
    print("0. Exit")
    print("="*60)


def main():
    """Main function"""
    print_banner()
    
    # CLI mode
    if len(sys.argv) > 1:
        run_cli()
        return
    
    # Interactive menu
    while True:
        show_menu()
        choice = input("\n👉 Your choice: ").strip()
        
        if choice == '1':
            app = PoseEstimatorApp()
            app.run_interactive()
        
        elif choice == '2':
            path = input("📁 Enter image path: ").strip()
            if Path(path).exists():
                estimator = YOLOv8PoseEstimator()
                estimator.predict_image(path)
            else:
                print("❌ File not found!")
        
        elif choice == '3':
            path = input("📁 Enter video path: ").strip()
            if Path(path).exists():
                estimator = YOLOv8PoseEstimator()
                estimator.predict_video(path)
            else:
                print("❌ File not found!")
        
        elif choice == '4':
            print("📹 Opening camera... (Press Q to exit)")
            estimator = YOLOv8PoseEstimator()
            estimator.predict_video(0)
        
        elif choice == '5':
            quick_test()
        
        elif choice == '6':
            print_help()
        
        elif choice == '7':
            Config.print_paths()
        
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
    
    # Process image
    python main.py --image path/to/image.jpg
    
    # Process video
    python main.py --video path/to/video.mp4
    
    # Use camera
    python main.py --camera
    
    # Specify model
    python main.py --model yolov8m-pose.pt --image test.jpg
    
    # Set confidence threshold
    python main.py --conf 0.7 --video test.mp4
    # Custom output path
    python main.py --image test.jpg --output custom/path/result.jpg
    
    # Do not save results
    python main.py --video test.mp4 --no-save
    
    # Do not show display
    python main.py --video test.mp4 --no-show
    
    # Show project paths
    python main.py --show-paths
    
    # Run in interactive mode
    python main.py --interactive
    
    
    🔹 Available Models:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    yolov8n-pose.pt  → Very Fast (Nano)
    yolov8s-pose.pt  → Fast (Small)
    yolov8m-pose.pt  → Medium
    yolov8l-pose.pt  → Large
    yolov8x-pose.pt  → Extra Large (Best Accuracy)
    
    ℹ️  Models are automatically downloaded to: models/
    
    
    🔹 Output Locations:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Images:  output/images/output_[filename]
    Videos:  output/videos/output_[filename]
    Camera:  output/videos/camera_output_[timestamp].mp4
    
    
    🔹 Keyboard Controls:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Q  → Exit video processing
    
    
    🔹 Folder Structure:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    yolo/
    ├── models/              ← Models auto-download here
    ├── input/
    │   ├── images/         ← Place input images here
    │   └── videos/         ← Place input videos here
    └── output/
        ├── images/         ← Processed images saved here
        └── videos/         ← Processed videos saved here
    
    
    🔹 Example Commands:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Run without parameters (interactive menu)
    python main.py
    
    # Process all images in a folder
    python batch_process.py --input-dir input/images/
    
    # Process distant match footage (soccer)
    python main.py --video matches/game1.mp4 --model yolov8m-pose.pt --conf 0.6
    
    # Quick camera capture
    python main.py --camera
    
    🔹 Get Help
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