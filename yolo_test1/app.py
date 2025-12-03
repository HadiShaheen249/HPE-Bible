"""
Application Interface for YOLOv8 Pose Estimator
Enhanced with configurable image tiling
"""

from pose_estimator import YOLOv8PoseEstimator
from config import Config
from pathlib import Path


class PoseEstimatorApp:
    """Class for managing the application"""
    
    def __init__(self):
        """Initialize the application"""
        print("=" * 60)
        print("🤸‍♂️  YOLOv8 Pose Estimation Application")
        print("=" * 60)
        
        # Create Estimator object
        self.estimator = YOLOv8PoseEstimator(
            model_name=Config.MODEL_NAME,
            conf_threshold=Config.CONFIDENCE_THRESHOLD
        )

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

    def change_tile_grid(self):
        """Change image/video tiling grid"""
        print("\n🌐 Current grid size:", Config.TILE_GRID)
        print("A larger grid (e.g., 4x4) improves distant player detection but takes longer.")
        try:
            rows = int(input("Enter number of rows (e.g., 2, 3, 4): ").strip())
            cols = int(input("Enter number of columns (e.g., 2, 3, 4): ").strip())
            if rows >= 1 and cols >= 1:
                Config.TILE_GRID = (rows, cols)
                print(f"✅ Tile grid updated to: {Config.TILE_GRID}")
            else:
                print("❌ Grid size must be at least 1x1")
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")

    def run_interactive(self):
        """Run the application in interactive mode"""
        while True:
            print("\n" + "=" * 60)
            print("📋 Main Menu:")
            print("=" * 60)
            print("1. Process Image")
            print("2. Process Video")
            print("3. Use Camera")
            print("4. Change Model")
            print("5. Change Confidence Threshold")
            print("6. Change Image Split Grid (Tiling)")
            print("0. Exit")
            print("=" * 60)
            
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
                print("\nAvailable Models:")
                print("1. yolov8n-pose.pt (Fast)")
                print("2. yolov8s-pose.pt (Medium)")
                print("3. yolov8m-pose.pt (Good)")
                print("4. yolov8l-pose.pt (Excellent)")
                print("5. yolov8x-pose.pt (Best Accuracy)")
                
                model_choice = input("Choose model number: ").strip()
                models = {
                    '1': 'yolov8n-pose.pt',
                    '2': 'yolov8s-pose.pt',
                    '3': 'yolov8m-pose.pt',
                    '4': 'yolov8l-pose.pt',
                    '5': 'yolov8x-pose.pt'
                }
                
                if model_choice in models:
                    Config.MODEL_NAME = models[model_choice]
                    print(f"🔄 Loading new model: {models[model_choice]}")
                    self.estimator = YOLOv8PoseEstimator(
                        model_name=Config.MODEL_NAME,
                        conf_threshold=Config.CONFIDENCE_THRESHOLD
                    )
                else:
                    print("❌ Invalid choice!")
            
            elif choice == '5':
                try:
                    threshold = float(input("Enter confidence threshold (0.0 - 1.0): ").strip())
                    if 0 <= threshold <= 1:
                        Config.CONFIDENCE_THRESHOLD = threshold
                        self.estimator.conf_threshold = threshold
                        print(f"✅ Confidence threshold changed to: {threshold}")
                    else:
                        print("❌ Value must be between 0 and 1")
                except ValueError:
                    print("❌ Invalid value!")

            elif choice == '6':
                self.change_tile_grid()
            
            elif choice == '0':
                print("\n👋 Thank you for using the application!")
                break
            
            else:
                print("❌ Invalid choice!")


if __name__ == "__main__":
    app = PoseEstimatorApp()
    app.run_interactive()