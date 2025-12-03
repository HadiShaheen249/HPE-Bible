# api/routes/__init__.py
"""
API Routes Package
"""

from . import rtmpose_routes
from . import yolo_test1_routes
from . import yolo_test2_routes

__all__ = [
    "rtmpose_routes",
    "yolo_test1_routes", 
    "yolo_test2_routes"
]