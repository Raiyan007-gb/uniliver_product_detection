import requests
import json
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import os

# API endpoint
url = "http://localhost:8000/detect-images"

# Image paths - add as many as you need
image_paths = [
    "dataset/test/images/clear14_aug_2.jpg",
    "dataset/test/images/doveconditioner3_aug_1.jpg",
    "dataset/test/images/doveconditioner10_aug_1.jpg"
]

# Prepare the data for the request
data = {
    'image_paths': image_paths,
    'return_images': 'true'
}

# Make the request
response = requests.post(url, data=data)

# Check if request was successful
if response.status_code == 200:
    result = response.json()
    output_dir = result['output_directory']
    
    print(f"Processing complete")
    print(f"Output directory: {output_dir}")
    print(f"Processed {result['processed_images']} images")
    print(f"Summary files:")
    print(f"  JSON: {result['summary_json']}")
    print(f"  Text: {result['summary_txt']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)