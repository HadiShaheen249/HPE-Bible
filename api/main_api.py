"""
HPE Bible - Unified FastAPI
Main API for RTMPose, YOLO Test1, and YOLO Test2
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import sys
from pathlib import Path

# Add project directories to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "RTMPose"))
sys.path.insert(0, str(BASE_DIR / "yolo_test1"))
sys.path.insert(0, str(BASE_DIR / "yolo_test2"))

# Import routes
from api.routes import  yolo_test1_routes, yolo_test2_routes

# Create FastAPI app
app = FastAPI(
    title="HPE Bible API",
    description="Unified API for Human Pose Estimation using RTMPose and YOLO models",
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

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint with API information
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HPE Bible API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; }
            .endpoint {
                background: #f8f9fa;
                padding: 10px;
                margin: 10px 0;
                border-left: 4px solid #007bff;
            }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 HPE Bible API</h1>
            <p>Welcome to the Human Pose Estimation API</p>
            
            <h2>📚 Available Projects:</h2>
            
            
            <div class="endpoint">
                <strong>YOLO Test 1</strong> - YOLO pose detection
                <br><a href="/api/yolo-test1/info">Info</a> | 
                <a href="/docs#/YOLO%20Test%201">Endpoints</a>
            </div>
            
            <div class="endpoint">
                <strong>YOLO Test 2</strong> - Advanced YOLO detection
                <br><a href="/api/yolo-test2/info">Info</a> | 
                <a href="/docs#/YOLO%20Test%202">Endpoints</a>
            </div>
            
            <h2>📖 Documentation:</h2>
            <ul>
                <li><a href="/docs">Swagger UI</a></li>
                <li><a href="/redoc">ReDoc</a></li>
            </ul>
            
            <h2>🔍 Health Check:</h2>
            <a href="/health">Check API Health</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
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
    """
    API information
    """
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