"""
HPE Bible - Main Entry Point
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def main():
    """Main entry point"""
    import uvicorn
    
    print("=" * 60)
    print("🚀 HPE Bible API - Starting Server...")
    print("=" * 60)
    print(f"📁 Root Directory: {ROOT_DIR}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"🏠 Homepage: http://localhost:8000")
    print("=" * 60)
    print("\n✨ Press CTRL+C to stop the server\n")
    
    # Set environment variable to suppress yolo_test2 model selection prompt
    os.environ['YOLO_TEST2_AUTO_SELECT'] = 'true'
    
    uvicorn.run(
        "api.main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)