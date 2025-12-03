"""
HPE Bible - Main Entry Point
Quick start script for the FastAPI server
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def main():
    """
    Main entry point for HPE Bible API
    """
    import uvicorn
    from api.main_api import app
    
    print("=" * 60)
    print("🚀 HPE Bible API - Starting Server...")
    print("=" * 60)
    print(f"📁 Root Directory: {ROOT_DIR}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"📚 ReDoc: http://localhost:8000/redoc")
    print("=" * 60)
    print("\n✨ Press CTRL+C to stop the server\n")
    
    # Run the server
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
        sys.exit(1)