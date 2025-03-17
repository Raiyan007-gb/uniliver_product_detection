# # fastapi_app.py
# import os
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# from typing import List
# import cv2
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO
# from collections import Counter
# import uvicorn
# import shutil
# import tempfile
# import yaml
# from tqdm import tqdm

# app = FastAPI(title="YOLOv8 Detection API")

# class YOLOv8DetectionPipeline:
#     def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45, img_size=640):
#         """
#         Initialize the YOLOv8 detection pipeline.
        
#         Args:
#             model_path (str): Path to the fine-tuned YOLOv8 model weights
#             conf_thresh (float): Confidence threshold for detections
#             iou_thresh (float): IoU threshold for non-maximum suppression
#             img_size (int): Image size for detection
#         """
#         self.conf_thresh = conf_thresh
#         self.iou_thresh = iou_thresh
#         self.img_size = img_size
        
#         # Load model
#         print(f"Loading YOLOv8 model from {model_path}")
#         self.model = YOLO(model_path)
        
#         # Extract class names directly from the model
#         self.class_names = self.model.names
        
#         # If class names weren't loaded from the model, try to load from external files
#         if not self.class_names:
#             print("Class names not found in model. Trying to load from external files...")
#             self.class_names = self._get_class_names(model_path)
            
#         print(f"Loaded {len(self.class_names)} class names: {self.class_names}")
        
#     def _get_class_names(self, model_path):
#         """Try to extract class names from the model directory."""
#         model_dir = Path(model_path).parent
#         yaml_path = model_dir / 'data.yaml'
        
#         if yaml_path.exists():
#             with open(yaml_path, 'r') as f:
#                 data = yaml.safe_load(f)
#                 if 'names' in data:
#                     return data['names']
        
#         txt_path = model_dir / 'classes.txt'
#         if txt_path.exists():
#             with open(txt_path, 'r') as f:
#                 return [line.strip() for line in f.readlines()]
        
#         print("Warning: Could not find class names. Using default numbering.")
#         return {i: f"Class {i}" for i in range(100)}
    
#     def process_image(self, image_path=None, output_path=None, visualize=True, image=None):
#         """
#         Process a single image with the YOLOv8 model.
#         """
#         if image is not None:
#             img = image
#         else:
#             img = cv2.imread(image_path)
            
#         if img is None:
#             print(f"Error: Could not process image")
#             return [], {}, 0.0
        
#         results = self.model(img, conf=self.conf_thresh, iou=self.iou_thresh, imgsz=self.img_size)
        
#         detections = []
#         class_counts = Counter()
#         total_confidence = 0.0
        
#         for result in results:
#             boxes = result.boxes
            
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
#                 conf = float(box.conf[0].cpu().numpy())
#                 cls_id = int(box.cls[0].cpu().numpy())
                
#                 class_name = self.class_names[cls_id]
#                 class_counts[class_name] += 1
#                 total_confidence += conf
                
#                 width = x2 - x1
#                 height = y2 - y1
                
#                 detections.append([class_name, conf, x1, y1, width, height])
        
#         avg_confidence = total_confidence / len(detections) if detections else 0.0
#         return detections, dict(class_counts), avg_confidence
    
#     def process_multiple_images(self, image_paths, output_dir=None, visualize=True, save_results=True):
#         """
#         Process multiple images.
#         """
#         results = {}
#         all_class_counts = {}
#         all_confidences = {}
#         total_counts = Counter()
#         class_confidences = {}
        
#         for img_path in tqdm(image_paths, desc="Processing images"):
#             img_path = Path(img_path)
#             detections, class_counts, avg_confidence = self.process_image(
#                 str(img_path),
#                 None,
#                 visualize
#             )
            
#             results[str(img_path)] = detections
#             all_class_counts[str(img_path)] = class_counts
#             all_confidences[str(img_path)] = avg_confidence
            
#             total_counts.update(class_counts)
            
#             for det in detections:
#                 class_name = det[0]
#                 conf = det[1]
#                 if class_name not in class_confidences:
#                     class_confidences[class_name] = []
#                 class_confidences[class_name].append(conf)
        
#         return results, all_class_counts, all_confidences
    
#     def get_accuracy_percentage(self, avg_confidence):
#         """
#         Convert average confidence to an accuracy percentage.
#         """
#         if avg_confidence > 0.8:
#             return 90 + (avg_confidence - 0.8) * 50
#         elif avg_confidence > 0.6:
#             return 80 + (avg_confidence - 0.6) * 50
#         elif avg_confidence > 0.4:
#             return 70 + (avg_confidence - 0.4) * 50
#         else:
#             return max(50, 50 + (avg_confidence - 0.25) * 133)
    
#     def get_accuracy_label(self, avg_confidence):
#         """
#         Convert average confidence to a human-readable label.
#         """
#         if avg_confidence > 0.8:
#             return "Excellent"
#         elif avg_confidence > 0.6:
#             return "Good"
#         elif avg_confidence > 0.4:
#             return "Moderate"
#         else:
#             return "Low"

# # Global pipeline instance
# pipeline = None

# @app.on_event("startup")
# async def startup_event():
#     global pipeline
#     # Update this with your actual model path
#     model_path = "./UBLModel_v2.pt"
#     pipeline = YOLOv8DetectionPipeline(
#         model_path=model_path,
#         conf_thresh=0.25,
#         iou_thresh=0.45,
#         img_size=640
#     )

# @app.post("/detect/")
# async def detect_objects(files: List[UploadFile] = File(...)):
#     """
#     Endpoint to detect objects in multiple images using YOLOv8
#     """
#     try:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             image_paths = []
#             results_dict = {}

#             # Save uploaded files temporarily
#             for file in files:
#                 temp_path = os.path.join(temp_dir, file.filename)
#                 with open(temp_path, "wb") as buffer:
#                     shutil.copyfileobj(file.file, buffer)
#                 image_paths.append(temp_path)

#             # Process images
#             results, class_counts, confidences = pipeline.process_multiple_images(
#                 image_paths=image_paths,
#                 output_dir=None,
#                 visualize=False,
#                 save_results=False
#             )

#             # Format response
#             response_data = {
#                 "detections": {},
#                 "summary": {
#                     "total_images": len(results),
#                     "overall_accuracy": 0.0,
#                     "detection_quality": "",
#                     "total_objects_by_class": {}
#                 }
#             }

#             # Fill detection results
#             for img_path, detections in results.items():
#                 img_name = Path(img_path).name
#                 response_data["detections"][img_name] = {
#                     "detections": [{
#                         "class_name": det[0],
#                         "confidence": float(det[1]),
#                         "bbox": {
#                             "x": int(det[2]),
#                             "y": int(det[3]),
#                             "width": int(det[4]),
#                             "height": int(det[5])
#                         }
#                     } for det in detections],
#                     "class_counts": class_counts[img_path],
#                     "average_confidence": float(confidences[img_path]),
#                     "estimated_accuracy": pipeline.get_accuracy_percentage(confidences[img_path]),
#                     "quality": pipeline.get_accuracy_label(confidences[img_path])
#                 }

#             # Calculate summary statistics
#             overall_avg_conf = sum(confidences.values()) / len(confidences) if confidences else 0.0
#             total_counts = Counter()
#             for counts in class_counts.values():
#                 total_counts.update(counts)

#             response_data["summary"]["overall_accuracy"] = pipeline.get_accuracy_percentage(overall_avg_conf)
#             response_data["summary"]["detection_quality"] = pipeline.get_accuracy_label(overall_avg_conf)
#             response_data["summary"]["total_objects_by_class"] = dict(total_counts)

#             return JSONResponse(content=response_data)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "model_loaded": pipeline is not None}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# fastapi_app.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import uvicorn
import shutil
import tempfile
import yaml
from tqdm import tqdm
from config import CONFIG  # Import the config

app = FastAPI(title=CONFIG.APP_TITLE)  # Use config title

class YOLOv8DetectionPipeline:
    def __init__(self, model_path=CONFIG.MODEL_PATH, 
                 conf_thresh=CONFIG.CONF_THRESHOLD, 
                 iou_thresh=CONFIG.IOU_THRESHOLD, 
                 img_size=CONFIG.IMG_SIZE):
        """
        Initialize the YOLOv8 detection pipeline.
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        
        print(f"Loading YOLOv8 model from {model_path}")
        self.model = YOLO(model_path)
        
        self.class_names = self.model.names
        
        if not self.class_names:
            print("Class names not found in model. Trying to load from external files...")
            self.class_names = self._get_class_names(model_path)
            if not self.class_names:
                self.class_names = CONFIG.DEFAULT_CLASS_NAMES
            
        print(f"Loaded {len(self.class_names)} class names: {self.class_names}")
        
    def _get_class_names(self, model_path):
        """Try to extract class names from the model directory."""
        model_dir = Path(model_path).parent
        yaml_path = model_dir / 'data.yaml'
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return data['names']
        
        txt_path = model_dir / 'classes.txt'
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        return None
    
    def process_image(self, image_path=None, output_path=None, visualize=True, image=None):
        if image is not None:
            img = image
        else:
            img = cv2.imread(image_path)
            
        if img is None:
            print(f"Error: Could not process image")
            return [], {}, 0.0
        
        results = self.model(img, conf=self.conf_thresh, iou=self.iou_thresh, imgsz=self.img_size)
        
        detections = []
        class_counts = Counter()
        total_confidence = 0.0
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                class_name = self.class_names[cls_id]
                class_counts[class_name] += 1
                total_confidence += conf
                
                width = x2 - x1
                height = y2 - y1
                
                detections.append([class_name, conf, x1, y1, width, height])
        
        avg_confidence = total_confidence / len(detections) if detections else 0.0
        return detections, dict(class_counts), avg_confidence
    
    def process_multiple_images(self, image_paths, output_dir=None, visualize=True, save_results=True):
        results = {}
        all_class_counts = {}
        all_confidences = {}
        total_counts = Counter()
        class_confidences = {}
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            img_path = Path(img_path)
            detections, class_counts, avg_confidence = self.process_image(
                str(img_path),
                None,
                visualize
            )
            
            results[str(img_path)] = detections
            all_class_counts[str(img_path)] = class_counts
            all_confidences[str(img_path)] = avg_confidence
            
            total_counts.update(class_counts)
            
            for det in detections:
                class_name = det[0]
                conf = det[1]
                if class_name not in class_confidences:
                    class_confidences[class_name] = []
                class_confidences[class_name].append(conf)
        
        return results, all_class_counts, all_confidences
    
    def get_accuracy_percentage(self, avg_confidence):
        if avg_confidence > 0.8:
            return 90 + (avg_confidence - 0.8) * 50
        elif avg_confidence > 0.6:
            return 80 + (avg_confidence - 0.6) * 50
        elif avg_confidence > 0.4:
            return 70 + (avg_confidence - 0.4) * 50
        else:
            return max(50, 50 + (avg_confidence - 0.25) * 133)
    
    def get_accuracy_label(self, avg_confidence):
        if avg_confidence > 0.8:
            return "Excellent"
        elif avg_confidence > 0.6:
            return "Good"
        elif avg_confidence > 0.4:
            return "Moderate"
        else:
            return "Low"

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = YOLOv8DetectionPipeline(
        model_path=CONFIG.MODEL_PATH,
        conf_thresh=CONFIG.CONF_THRESHOLD,
        iou_thresh=CONFIG.IOU_THRESHOLD,
        img_size=CONFIG.IMG_SIZE
    )

@app.post("/detect/")
async def detect_objects(files: List[UploadFile] = File(...)):
    try:
        with tempfile.TemporaryDirectory(prefix=CONFIG.TEMP_DIR_PREFIX) as temp_dir:
            image_paths = []
            results_dict = {}

            for file in files:
                temp_path = os.path.join(temp_dir, file.filename)
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                image_paths.append(temp_path)

            results, class_counts, confidences = pipeline.process_multiple_images(
                image_paths=image_paths,
                output_dir=None,
                visualize=False,
                save_results=False
            )

            response_data = {
                "detections": {},
                "summary": {
                    "total_images": len(results),
                    "overall_accuracy": 0.0,
                    "detection_quality": "",
                    "total_objects_by_class": {}
                }
            }

            for img_path, detections in results.items():
                img_name = Path(img_path).name
                response_data["detections"][img_name] = {
                    "detections": [{
                        "class_name": det[0],
                        "confidence": float(det[1]),
                        "bbox": {
                            "x": int(det[2]),
                            "y": int(det[3]),
                            "width": int(det[4]),
                            "height": int(det[5])
                        }
                    } for det in detections],
                    "class_counts": class_counts[img_path],
                    "average_confidence": float(confidences[img_path]),
                    "estimated_accuracy": pipeline.get_accuracy_percentage(confidences[img_path]),
                    "quality": pipeline.get_accuracy_label(confidences[img_path])
                }

            overall_avg_conf = sum(confidences.values()) / len(confidences) if confidences else 0.0
            total_counts = Counter()
            for counts in class_counts.values():
                total_counts.update(counts)

            response_data["summary"]["overall_accuracy"] = pipeline.get_accuracy_percentage(overall_avg_conf)
            response_data["summary"]["detection_quality"] = pipeline.get_accuracy_label(overall_avg_conf)
            response_data["summary"]["total_objects_by_class"] = dict(total_counts)

            return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": pipeline is not None,
        "current_model": CONFIG.current_model,
        "model_path": str(CONFIG.MODEL_PATH)
    }

@app.post("/switch-model/{model_key}")
async def switch_model(model_key: str):
    """Switch to a different model configuration"""
    try:
        global pipeline
        new_model_path = CONFIG.switch_model(model_key)
        pipeline = YOLOv8DetectionPipeline(
            model_path=new_model_path,
            conf_thresh=CONFIG.CONF_THRESHOLD,
            iou_thresh=CONFIG.IOU_THRESHOLD,
            img_size=CONFIG.IMG_SIZE
        )
        return {"status": "success", "message": f"Switched to model: {model_key}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.HOST, port=CONFIG.PORT)