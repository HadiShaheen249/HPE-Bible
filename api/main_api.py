"""
HPE Bible - Unified FastAPI
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

# Add project directories to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "yolo_test1"))
sys.path.insert(0, str(BASE_DIR / "yolo_test2"))

# Import routes
from api.routes import yolo_test1_routes, yolo_test2_routes

# Create FastAPI app
app = FastAPI(
    title="HPE Bible API",
    description="Unified API for Human Pose Estimation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    yolo_test1_routes.router,
    prefix="/api/yolo-test1",
    tags=["YOLO Test 1"]
)
app.include_router(
    yolo_test2_routes.router,
    prefix="/api/yolo-test2",
    tags=["YOLO Test 2"]
)

# Home page with beautiful UI
@app.get("/", response_class=HTMLResponse)
async def home():
    """Beautiful home page"""
    html_file = Path(__file__).parent / "templates" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(content="<h1>Template not found</h1>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "projects": {
            "yolo_test1": "available",
            "yolo_test2": "available"
        }
    }

@app.get("/api/info")
async def api_info():
    """API information"""
    return {
        "name": "HPE Bible API",
        "version": "1.0.0",
        "description": "Unified API for pose estimation",
        "endpoints": {
            "yolo_test1": "/api/yolo-test1",
            "yolo_test2": "/api/yolo-test2"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting HPE Bible API...")
    print("📖 Documentation: http://localhost:8000/docs")
    print("🏠 Homepage: http://localhost:8000")
    
    uvicorn.run(
        "api.main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )