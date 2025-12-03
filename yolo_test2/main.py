"""
Main entry point for Football Pose Estimation Project
Provides command-line interface and easy-to-use functions
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Union

# Import all modules
try:
    from config import (
        INPUT_DIR,
        OUTPUT_DIR,
        LOGGING_CONFIG,
        print_config_summary
    )
    from models import initialize_models
    from detector import PlayerDetector
    from pose_estimator import PoseEstimator
    from video_processor import VideoProcessor, ImageProcessor, BatchProcessor
    from utils import is_video_file, is_image_file
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("Make sure all required files are present")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["log_level"]),
    format=LOGGING_CONFIG["log_format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PoseEstimationApp:
    """
    Main application class for pose estimation
    """
    
    def __init__(self):
        """Initialize the application"""
        self.detector_model = None
        self.pose_model = None
        self.model_manager = None
        self.player_detector = None
        self.pose_estimator = None
        self.video_processor = None
        self.image_processor = None
        self.batch_processor = None
        
        self.initialized = False
    
    def initialize(self):
        """Initialize all models and processors"""
        if self.initialized:
            logger.info("⚠️  Application is already initialized")
            return
        
        try:
            logger.info("\n" + "="*60)
            logger.info("🚀 Initializing Pose Estimation Application")
            logger.info("="*60)
            
            # Initialize models
            logger.info("\n📥 Loading models...")
            self.detector_model, self.pose_model, self.model_manager = initialize_models()
            
            # Create detector and pose estimator
            logger.info("\n🔧 Initializing processors...")
            self.player_detector = PlayerDetector(self.detector_model)
            self.pose_estimator = PoseEstimator(self.pose_model)
            
            # Create processors
            self.video_processor = VideoProcessor(
                self.player_detector, 
                self.pose_estimator,
                enable_tracking=True
            )
            self.image_processor = ImageProcessor(
                self.player_detector,
                self.pose_estimator
            )
            self.batch_processor = BatchProcessor(
                self.player_detector,
                self.pose_estimator,
                enable_tracking=True
            )
            
            self.initialized = True
            
            logger.info("\n" + "="*60)
            logger.info("✅ Application initialized successfully!")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"\n❌ Initialization failed: {str(e)}")
            raise
    
    def process_video(self, video_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     save_data: bool = True,
                     show_preview: bool = False):
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            save_data: Whether to save pose data
            show_preview: Whether to show live preview
        """
        if not self.initialized:
            self.initialize()
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if not is_video_file(video_path):
            raise ValueError(f"The file is not a valid video: {video_path}")
        
        return self.video_processor.process_video(
            video_path, output_path, save_data, show_preview
        )
    
    def process_image(self, image_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     save_data: bool = True):
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)
            save_data: Whether to save pose data
        """
        if not self.initialized:
            self.initialize()
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if not is_image_file(image_path):
            raise ValueError(f"The file is not a valid image: {image_path}")
        
        return self.image_processor.process_image(
            image_path, output_path, save_data
        )
    
    def process_directory(self, input_dir: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         process_videos: bool = True,
                         process_images: bool = True):
        """
        Process all files in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path (optional)
            process_videos: Whether to process videos
            process_images: Whether to process images
        """
        if not self.initialized:
            self.initialize()
        
        input_dir = Path(input_dir)
        
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Directory not found: {input_dir}")
        
        return self.batch_processor.process_directory(
            input_dir, output_dir, process_videos, process_images
        )
    
    def process_file(self, file_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     save_data: bool = True,
                     show_preview: bool = False):
        """
        Automatically detect and process file (video or image)
        
        Args:
            file_path: Path to input file
            output_path: Path to output file (optional)
            save_data: Whether to save data
            show_preview: Whether to show preview (for videos)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if is_video_file(file_path):
            logger.info("🎥 Video detected")
            return self.process_video(file_path, output_path, save_data, show_preview)
        elif is_image_file(file_path):
            logger.info("🖼️  Image detected")
            return self.process_image(file_path, output_path, save_data)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")


def main():
    """Main function with command-line interface"""
    
    # Print banner
    print_banner()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="⚽ Football Pose Estimation - Analyze player poses in football matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Process a video
  python main.py --video input/match.mp4
  
  # Process an image
  python main.py --image input/player.jpg
  
  # Auto process a file
  python main.py --file input/match.mp4
  
  # Process an entire directory
  python main.py --directory input/
  
  # Process with live preview
  python main.py --video input/match.mp4 --preview
  
  # Process without saving data
  python main.py --video input/match.mp4 --no-save
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '--video', '-v',
        type=str,
        help='Path to video file for processing'
    )
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to image file for processing'
    )
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Path to file (type will be auto-detected)'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Path to directory to process all files'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file/directory path (optional)',
        default=None
    )
    
    # Processing options
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save pose data (JSON/CSV)'
    )
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Show live preview for videos during processing'
    )
    parser.add_argument(
        '--no-videos',
        action='store_true',
        help='Skip video processing when processing a directory'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip image processing when processing a directory'
    )
    
    # Interactive mode
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive or (not any([args.video, args.image, args.file, args.directory])):
        run_interactive_mode()
        return
    
    try:
        # Create app instance
        app = PoseEstimationApp()
        
        # Process based on arguments
        if args.video:
            app.process_video(
                args.video,
                args.output,
                save_data=not args.no_save,
                show_preview=args.preview
            )
        
        elif args.image:
            app.process_image(
                args.image,
                args.output,
                save_data=not args.no_save
            )
        
        elif args.file:
            app.process_file(
                args.file,
                args.output,
                save_data=not args.no_save,
                show_preview=args.preview
            )
        
        elif args.directory:
            app.process_directory(
                args.directory,
                args.output,
                process_videos=not args.no_videos,
                process_images=not args.no_images
            )
        
        logger.info("\n✅ Processing completed successfully!")

    except KeyboardInterrupt:
            logger.info("\n\n⚠️ Processing was stopped by the user")
            sys.exit(0)

    except Exception as e:
            logger.error(f"\n❌ An error occurred: {str(e)}")
            sys.exit(1)


def run_interactive_mode():
    """Run interactive mode with menu"""
    
    print("\n" + "="*60)
    print("🎮 Interactive Mode")
    print("="*60)
    
    # Create app instance
    app = PoseEstimationApp()
    
    while True:
        print("\n📋 Main Menu:")
        print("  [1] Process Video")
        print("  [2] Process Image")
        print("  [3] Auto Process File")
        print("  [4] Process Directory")
        print("  [5] Show Settings")
        print("  [0] Exit")
        print()
        
        choice = input("Select an option by number: ").strip()
        
        try:
            if choice == '1':
                process_video_interactive(app)
            
            elif choice == '2':
                process_image_interactive(app)
            
            elif choice == '3':
                process_file_interactive(app)
            
            elif choice == '4':
                process_directory_interactive(app)
            
            elif choice == '5':
                print_config_summary()
                if app.initialized:
                    app.model_manager.print_model_info()
            
            elif choice == '0':
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice, please try again")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled")
            continue
        
        except Exception as e:
            logger.error(f"\n❌ An error occurred: {str(e)}")
            continue


def process_video_interactive(app: PoseEstimationApp):
    """Interactive video processing"""
    print("\n" + "="*60)
    print("🎥 Process Video")
    print("="*60)
    
    # Get video path
    video_path = input("\nEnter the video path: ").strip()
    
    if not video_path:
        print("❌ No path entered")
        return
    
    # Get output path
    output_path = input("Enter the output path (leave empty for default): ").strip()
    if not output_path:
        output_path = None
    
    # Ask for preview
    preview = input("Show live preview? (y/n) [n]: ").strip().lower() == 'y'
    
    # Ask for data saving
    save_data = input("Save pose data? (y/n) [y]: ").strip().lower() != 'n'
    
    # Process
    print("\n⏳ Processing...")
    result = app.process_video(video_path, output_path, save_data, preview)
    
    print(f"\n✅ Processing completed!")
    print(f"📤 Output file: {result['output_path']}")


def process_image_interactive(app: PoseEstimationApp):
    """Interactive image processing"""
    print("\n" + "="*60)
    print("🖼️  Process Image")
    print("="*60)
    
    # Get image path
    image_path = input("\nEnter the image path: ").strip()
    
    if not image_path:
        print("❌ No path entered")
        return
    
    # Get output path
    output_path = input("Enter the output path (leave empty for default): ").strip()
    if not output_path:
        output_path = None
    
    # Ask for data saving
    save_data = input("Save pose data? (y/n) [y]: ").strip().lower() != 'n'
    
    # Process
    print("\n⏳ Processing...")
    result = app.process_image(image_path, output_path, save_data)
    
    print(f"\n✅ Processing completed!")
    print(f"📤 Output file: {result['output_path']}")
    print(f"👥 Number of players: {result['num_detections']}")
    print(f"🧍 Number of poses: {result['num_poses']}")


def process_file_interactive(app: PoseEstimationApp):
    """Interactive file processing (auto-detect)"""
    print("\n" + "="*60)
    print("📄 Auto Process File")
    print("="*60)
    
    # Get file path
    file_path = input("\nEnter the file path: ").strip()
    
    if not file_path:
        print("❌ No path entered")
        return
    
    # Get output path
    output_path = input("Enter the output path (leave empty for default): ").strip()
    if not output_path:
        output_path = None
    
    # Ask for preview (for videos)
    preview = input("Show live preview? (videos only) (y/n) [n]: ").strip().lower() == 'y'
    
    # Ask for data saving
    save_data = input("Save pose data? (y/n) [y]: ").strip().lower() != 'n'
    
    # Process
    print("\n⏳ Processing...")
    result = app.process_file(file_path, output_path, save_data, preview)
    
    print(f"\n✅ Processing completed!")
    print(f"📤 Output file: {result['output_path']}")


def process_directory_interactive(app: PoseEstimationApp):
    """Interactive directory processing"""
    print("\n" + "="*60)
    print("📁 Process Directory")
    print("="*60)
    
    # Get directory path
    dir_path = input("\nEnter the directory path: ").strip()
    
    if not dir_path:
        print("❌ No path entered")
        return
    
    # Get output path
    output_path = input("Enter the output directory path (leave empty for default): ").strip()
    if not output_path:
        output_path = None
    
    # Ask what to process
    process_videos = input("Process videos? (y/n) [y]: ").strip().lower() != 'n'
    process_images = input("Process images? (y/n) [y]: ").strip().lower() != 'n'
    
    # Process
    print("\n⏳ Processing...")
    result = app.process_directory(dir_path, output_path, process_videos, process_images)
    
    print(f"\n✅ Processing completed!")
    print(f"📤 Output directory: {result['output_dir']}")


def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        ⚽ Football Pose Estimation System ⚽              ║
    ║                                                          ║
    ║          Analyze player poses in football matches       ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    📁 Input directory: {INPUT_DIR}")
    print(f"    📤 Output directory: {OUTPUT_DIR}")
    print()


# Simple usage functions for direct import
def process_video(video_path: str, output_path: str = None):
    """
    Simple function to process a video
    
    Args:
        video_path: Path to video file
        output_path: Output path (optional)
    """
    app = PoseEstimationApp()
    return app.process_video(video_path, output_path)


def process_image(image_path: str, output_path: str = None):
    """
    Simple function to process an image
    
    Args:
        image_path: Path to image file
        output_path: Output path (optional)
    """
    app = PoseEstimationApp()
    return app.process_image(image_path, output_path)


if __name__ == "__main__":
    main()
