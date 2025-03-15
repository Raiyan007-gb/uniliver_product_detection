import requests
import json
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import os

# API endpoint
url = "http://localhost:8000/detect-folder"

# Folder path containing images
folder_path = "C:/Users/RDR/Documents/Unilever/UBLvsNON-UBL"

# Prepare the data for the request
data = {
    'folder_path': folder_path,
    'return_images': 'true'
}

# Make the request
response = requests.post(url, data=data)

# Check if request was successful
if response.status_code == 200:
    results = response.json()['results']
    
    print(f"Processed {len(results)} images from folder")
    
    # Loop through results for each image
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {os.path.basename(result['image_path'])}")
        print(f"Accuracy: {result['accuracy']:.2f}%")
        print(f"Quality: {result['quality']}")
        
        # Print detected objects
        print("Detected objects:")
        for class_name, count in result['class_counts'].items():
            print(f"  {class_name}: {count}")
        

else:
    print(f"Error: {response.status_code}")
    print(response.text)