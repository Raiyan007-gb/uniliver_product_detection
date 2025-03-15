import os
import io
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel
from pathlib import Path
import time
import tempfile
import json
# Import your existing YOLOv8 detection pipeline
from detection_pipeline import YOLOv8DetectionPipeline
# Import configuration
import config

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]  # [x, y, width, height]

class ImageDetectionResponse(BaseModel):
    image_name: str
    detections: List[DetectionResult]
    class_counts: dict
    accuracy: float
    quality: str
    base64_image: Optional[str] = None

# Initialize the model at startup
detection_pipeline = None

@app.on_event("startup")
async def startup_event():
    global detection_pipeline
    print(f"Loading YOLOv8 model from {config.MODEL_PATH}")
    detection_pipeline = YOLOv8DetectionPipeline(
        model_path=config.MODEL_PATH,
        conf_thresh=config.CONF_THRESH,
        iou_thresh=config.IOU_THRESH,
        img_size=config.IMG_SIZE
    )
    print("Model loaded successfully")

@app.get("/")
async def root():
    return {"message": "YOLOv8 Detection API is running"}

@app.get("/health")
async def health_check():
    if detection_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": config.MODEL_PATH}

@app.get("/model/info")
async def model_info():
    if detection_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_path": config.MODEL_PATH,
        "confidence_threshold": config.CONF_THRESH,
        "iou_threshold": config.IOU_THRESH,
        "image_size": config.IMG_SIZE,
        "classes": detection_pipeline.class_names
    }

def read_image_file(file) -> np.ndarray:
    """Read image file to numpy array"""
    image = Image.open(io.BytesIO(file))
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Convert to numpy array
    image_np = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_np

def encode_image_to_base64(image_np):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/detect", response_model=ImageDetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    return_image: bool = Form(False)
):
    """
    Detect objects in an uploaded image
    
    - **file**: Image file to process
    - **return_image**: If True, include annotated image in base64 format in the response
    """
    if detection_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read the image
        contents = await file.read()
        image = read_image_file(contents)
        
        # Create base output directory and timestamped subdirectory
        base_output_dir = os.path.join(os.getcwd(), "detection_output")
        os.makedirs(base_output_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Set output path for the detected image
        if return_image:
            temp_output_path = os.path.join(output_dir, f"{file.filename}")
        else:
            temp_output_path = None
        
        # Process the image
        detections, class_counts, avg_confidence = detection_pipeline.process_image(
            image=image,
            output_path=temp_output_path,
            visualize=return_image
        )
        
        # Prepare response
        detection_results = []
        for det in detections:
            class_name, confidence, x, y, width, height = det
            detection_results.append(
                DetectionResult(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[x, y, width, height]
                )
            )
        
        # Get quality label and accuracy percentage
        quality = detection_pipeline.get_accuracy_label(avg_confidence)
        accuracy = detection_pipeline.get_accuracy_percentage(avg_confidence)
        
        # Prepare the response
        response = {
            "image_name": file.filename,
            "detections": detection_results,
            "class_counts": class_counts,
            "accuracy": accuracy,
            "quality": quality,
            "base64_image": None
        }
        
        # Add the annotated image if requested
        if return_image:
            try:
                output_image = cv2.imread(temp_output_path)
                response["base64_image"] = encode_image_to_base64(output_image)
            except Exception as e:
                print(f"Error encoding output image: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
        
        return response
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-folder")
async def detect_folder(
    folder_path: str = Form(...),
    return_images: bool = Form(True)  # Changed default to True
):
    """
    Detect objects in all images within a specified folder
    
    - **folder_path**: Path to the folder containing images
    - **return_images**: If True, include annotated images in base64 format in the response
    """
    if detection_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Check if folder exists
        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")
        
        # Create base output directory
        base_output_dir = os.path.join(os.getcwd(), "detection_output")
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create timestamped subdirectory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the folder and save detected images
        results, class_counts, confidences = detection_pipeline.process_directory(
            input_dir=folder_path,
            output_dir=output_dir,
            visualize=True,  # Enable visualization to save detected images
            save_results=True  # Save the detected images
        )
        
        # Prepare and save summary report
        summary_report = []
        processed_images = []
        
        for img_path, detections in results.items():
            img_class_counts = class_counts.get(img_path, {})
            avg_confidence = confidences.get(img_path, 0.0)
            quality = detection_pipeline.get_accuracy_label(avg_confidence)
            accuracy = detection_pipeline.get_accuracy_percentage(avg_confidence)
            
            summary = {
                "image_path": img_path,
                "detections_count": len(detections),
                "class_counts": img_class_counts,
                "accuracy": accuracy,
                "quality": quality,
                "detections": [
                    {
                        "class_name": det[0],
                        "confidence": det[1],
                        "bbox": det[2:]
                    } for det in detections
                ]
            }
            summary_report.append(summary)
            
            # Store processed image paths
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            if os.path.exists(out_path):
                processed_images.append(out_path)
        
        # Save summary report
        
        # Save JSON summary
        summary_json = f"detection_summary_{timestamp}.json"
        with open(os.path.join(output_dir, summary_json), "w") as f:
            json.dump(summary_report, f, indent=4)
        
        # Save TXT summary
        summary_txt = f"detection_summary_{timestamp}.txt"
        with open(os.path.join(output_dir, summary_txt), "w") as f:
            f.write(f"Detection Summary Report - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            for summary in summary_report:
                f.write(f"Image: {os.path.basename(summary['image_path'])}\n")
                f.write(f"Detection Quality: {summary['quality']}\n")
                f.write(f"Accuracy: {summary['accuracy']:.2f}%\n")
                f.write(f"Total Detections: {summary['detections_count']}\n\n")
                
                f.write("Objects Detected:\n")
                for class_name, count in summary['class_counts'].items():
                    f.write(f"  {class_name}: {count}\n")
                f.write("\n" + "-" * 30 + "\n\n")
        
        return {
            "message": "Processing complete",
            "output_directory": output_dir,
            "processed_images": len(processed_images),
            "summary_json": os.path.join(output_dir, summary_json),
            "summary_txt": os.path.join(output_dir, summary_txt)
        }
        
    except Exception as e:
        print(f"Error processing folder: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing folder: {str(e)}")
@app.post("/detect-images")
async def detect_images(
    image_paths: List[str] = Form(...),
    return_images: bool = Form(True)
):
    """
    Detect objects in multiple images specified by their paths
    
    - **image_paths**: List of paths to the images to process
    - **return_images**: If True, include annotated images in base64 format in the response
    """
    if detection_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate image paths
        valid_paths = []
        for path in image_paths:
            if not os.path.isfile(path):
                print(f"Warning: {path} is not a valid file. Skipping.")
                continue
            valid_paths.append(path)
        
        if not valid_paths:
            raise HTTPException(status_code=400, detail="No valid image paths provided")
        
        # Create base output directory
        base_output_dir = os.path.join(os.getcwd(), "detection_output")
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create timestamped subdirectory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the images and save detected images
        results, class_counts, confidences = detection_pipeline.process_multiple_images(
            image_paths=valid_paths,
            output_dir=output_dir,
            visualize=True,
            save_results=True
        )
        
        # Prepare and save summary report
        summary_report = []
        processed_images = []
        
        for img_path, detections in results.items():
            img_class_counts = class_counts.get(img_path, {})
            avg_confidence = confidences.get(img_path, 0.0)
            quality = detection_pipeline.get_accuracy_label(avg_confidence)
            accuracy = detection_pipeline.get_accuracy_percentage(avg_confidence)
            
            summary = {
                "image_path": img_path,
                "detections_count": len(detections),
                "class_counts": img_class_counts,
                "accuracy": accuracy,
                "quality": quality,
                "detections": [
                    {
                        "class_name": det[0],
                        "confidence": det[1],
                        "bbox": det[2:]
                    } for det in detections
                ]
            }
            summary_report.append(summary)
            
            # Store processed image paths
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            if os.path.exists(out_path):
                processed_images.append(out_path)
        
        # Save JSON summary
        summary_json = f"detection_summary_{timestamp}.json"
        with open(os.path.join(output_dir, summary_json), "w") as f:
            json.dump(summary_report, f, indent=4)
        
        # Save TXT summary
        summary_txt = f"detection_summary_{timestamp}.txt"
        with open(os.path.join(output_dir, summary_txt), "w") as f:
            f.write(f"Detection Summary Report - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            for summary in summary_report:
                f.write(f"Image: {os.path.basename(summary['image_path'])}\n")
                f.write(f"Detection Quality: {summary['quality']}\n")
                f.write(f"Accuracy: {summary['accuracy']:.2f}%\n")
                f.write(f"Total Detections: {summary['detections_count']}\n\n")
                
                f.write("Objects Detected:\n")
                for class_name, count in summary['class_counts'].items():
                    f.write(f"  {class_name}: {count}\n")
                f.write("\n" + "-" * 30 + "\n\n")
        
        return {
            "message": "Processing complete",
            "output_directory": output_dir,
            "processed_images": len(processed_images),
            "summary_json": os.path.join(output_dir, summary_json),
            "summary_txt": os.path.join(output_dir, summary_txt)
        }
        
    except Exception as e:
        print(f"Error processing images: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
if __name__ == "__main__":
    # Run the server using configuration from config.py
    uvicorn.run(
        "api:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )