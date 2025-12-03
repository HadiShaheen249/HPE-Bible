"""
Application Interface for RTMPose + ByteTrack Pose Estimator
"""

from pose_estimator import RTMPoseEstimator
from config import Config
from pathlib import Path
import sys


class PoseEstimatorApp:
    """Class for managing the application"""
    
    def __init__(self):
        """Initialize the application"""
        print("="*60)
        print("🤸‍♂️  RTMPose + ByteTrack Pose Estimation Application")
        print("="*60)
        
        # Create Estimator object
        self.estimator = None
        self.det_conf = Config.DET_CONFIDENCE
        self.pose_conf = Config.POSE_CONFIDENCE
        self.device = Config.DEVICE
        self.tracking_enabled = True
        
        self._init_estimator()
    
    def _init_estimator(self):
        """Initialize the estimator"""
        try:
            print(f"\n🔄 Initializing estimator...")
            self.estimator = RTMPoseEstimator(
                det_conf=self.det_conf,
                pose_conf=self.pose_conf,
                device=self.device
            )
            
            if not self.tracking_enabled:
                self.estimator.tracker = None
                print("⚠️  Tracking disabled")
            
        except Exception as e:
            print(f"❌ Error initializing estimator: {e}")
            print(f"\n💡 Make sure you have installed:")
            print(f"   pip install openmim")
            print(f"   mim install mmdet mmpose")
            sys.exit(1)
    
    def process_image(self, image_path: str):
        """Process a single image"""
        try:
            self.estimator.predict_image(
                image_path=image_path,
                save_result=Config.SAVE_RESULTS
            )
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    
    def process_video(self, video_path: str):
        """Process a video file"""
        try:
            self.estimator.predict_video(
                video_path=video_path,
                save_result=Config.SAVE_RESULTS,
                show_live=Config.SHOW_LIVE
            )
        except Exception as e:
            print(f"❌ Error processing video: {e}")
    
    def process_camera(self):
        """Process camera stream"""
        try:
            self.estimator.predict_video(
                video_path=0,
                save_result=Config.SAVE_RESULTS,
                show_live=True
            )
        except Exception as e:
            print(f"❌ Error opening camera: {e}")
    
    def change_detection_confidence(self):
        """Change detection confidence threshold"""
        print(f"\n📊 Current detection confidence: {self.det_conf}")
        try:
            new_conf = float(input("Enter new confidence (0.0 - 1.0): ").strip())
            if 0 <= new_conf <= 1:
                self.det_conf = new_conf
                self.estimator.det_conf = new_conf
                print(f"✅ Detection confidence changed to: {new_conf}")
            else:
                print("❌ Value must be between 0 and 1")
        except ValueError:
            print("❌ Invalid value!")
    
    def change_pose_confidence(self):
        """Change pose confidence threshold"""
        print(f"\n📊 Current pose confidence: {self.pose_conf}")
        try:
            new_conf = float(input("Enter new confidence (0.0 - 1.0): ").strip())
            if 0 <= new_conf <= 1:
                self.pose_conf = new_conf
                self.estimator.pose_conf = new_conf
                print(f"✅ Pose confidence changed to: {new_conf}")
            else:
                print("❌ Value must be between 0 and 1")
        except ValueError:
            print("❌ Invalid value!")
    
    def toggle_tracking(self):
        """Toggle tracking on/off"""
        self.tracking_enabled = not self.tracking_enabled
        
        if self.tracking_enabled:
            if self.estimator.tracker is None:
                self.estimator._init_tracker()
            print("✅ Tracking enabled")
        else:
            self.estimator.tracker = None
            print("⚠️  Tracking disabled")
    
    def change_device(self):
        """Change computing device"""
        print(f"\n💻 Current device: {self.device}")
        print("Available devices:")
        print("1. cuda:0 (GPU)")
        print("2. cpu")
        
        choice = input("Choose device: ").strip()
        
        if choice == '1':
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                print("✅ Switched to GPU")
            else:
                print("❌ CUDA not available!")
                return
        elif choice == '2':
            self.device = 'cpu'
            print("✅ Switched to CPU")
        else:
            print("❌ Invalid choice!")
            return
        
        # Reinitialize estimator with new device
        print("🔄 Reinitializing estimator...")
        self._init_estimator()
    
    def show_settings(self):
        """Show current settings"""
        print("\n" + "="*60)
        print("⚙️  Current Settings:")
        print("="*60)
        print(f"Detection Model: {self.estimator.det_model_name}")
        print(f"Pose Model: {self.estimator.pose_model_name}")
        print(f"Detection Confidence: {self.det_conf}")
        print(f"Pose Confidence: {self.pose_conf}")
        print(f"Device: {self.device}")
        print(f"Tracking: {'Enabled' if self.tracking_enabled else 'Disabled'}")
        print(f"Save Results: {Config.SAVE_RESULTS}")
        print(f"Show Live: {Config.SHOW_LIVE}")
        print("="*60)
    
    def batch_process_images(self):
        """Batch process all images in input folder"""
        from utils import FileManager
        
        images = FileManager.get_all_images(Config.INPUT_IMAGES_DIR)
        
        if len(images) == 0:
            print(f"❌ No images found in: {Config.INPUT_IMAGES_DIR}")
            return
        
        print(f"\n📁 Found {len(images)} image(s)")
        confirm = input("Process all images? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("❌ Cancelled")
            return
        
        print("\n🔄 Processing images...")
        for idx, img_path in enumerate(images, 1):
            print(f"\n[{idx}/{len(images)}] Processing: {img_path.name}")
            try:
                self.process_image(str(img_path))
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print(f"\n✅ Batch processing completed!")
    
    def batch_process_videos(self):
        """Batch process all videos in input folder"""
        from utils import FileManager
        
        videos = FileManager.get_all_videos(Config.INPUT_VIDEOS_DIR)
        
        if len(videos) == 0:
            print(f"❌ No videos found in: {Config.INPUT_VIDEOS_DIR}")
            return
        
        print(f"\n📁 Found {len(videos)} video(s)")
        confirm = input("Process all videos? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("❌ Cancelled")
            return
        
        print("\n🔄 Processing videos...")
        for idx, vid_path in enumerate(videos, 1):
            print(f"\n[{idx}/{len(videos)}] Processing: {vid_path.name}")
            try:
                self.process_video(str(vid_path))
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print(f"\n✅ Batch processing completed!")
    
    def run_interactive(self):
        """Run the application in interactive mode"""
        while True:
            print("\n" + "="*60)
            print("📋 Main Menu:")
            print("="*60)
            print("1.  Process Image")
            print("2.  Process Video")
            print("3.  Use Camera")
            print("4.  Batch Process Images")
            print("5.  Batch Process Videos")
            print("6.  Change Detection Confidence")
            print("7.  Change Pose Confidence")
            print("8.  Toggle Tracking (On/Off)")
            print("9.  Change Device (GPU/CPU)")
            print("10. Show Current Settings")
            print("11. Show Paths")
            print("0.  Exit")
            print("="*60)
            
            choice = input("\n👉 Choose an option: ").strip()
            
            if choice == '1':
                path = input("📁 Enter image path: ").strip()
                if Path(path).exists():
                    self.process_image(path)
                else:
                    print("❌ File not found!")
            
            elif choice == '2':
                path = input("📁 Enter video path: ").strip()
                if Path(path).exists():
                    self.process_video(path)
                else:
                    print("❌ File not found!")
            
            elif choice == '3':
                print("📹 Opening camera... (Press Q to exit)")
                self.process_camera()
            
            elif choice == '4':
                self.batch_process_images()
            
            elif choice == '5':
                self.batch_process_videos()
            
            elif choice == '6':
                self.change_detection_confidence()
            
            elif choice == '7':
                self.change_pose_confidence()
            
            elif choice == '8':
                self.toggle_tracking()
            
            elif choice == '9':
                self.change_device()
            
            elif choice == '10':
                self.show_settings()
            
            elif choice == '11':
                Config.print_paths()
                Config.print_model_info()
            
            elif choice == '0':
                print("\n👋 Thank you for using the application!")
                break
            
            else:
                print("❌ Invalid choice!")
            
            input("\n⏸️  Press Enter to continue...")


if __name__ == "__main__":
    app = PoseEstimatorApp()
    app.run_interactive()