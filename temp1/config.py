# config.py
import os

# Model configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "./C:/Users/RDR/Documents/Unilever/ublVSnon_ublModel_v1.pt")  # Default model, override with env var
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.25"))
IOU_THRESH = float(os.environ.get("IOU_THRESH", "0.45"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))

# Server configuration
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# API configuration
API_TITLE = "YOLOv8 Detection API"
API_DESCRIPTION = "API for object detection using YOLOv8"
API_VERSION = "1.0.0"