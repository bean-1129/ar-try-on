from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import requests
from io import BytesIO
import base64
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load YOLO pose model once at startup
print("Loading YOLO pose model...")
pose_model = YOLO('yolov8n-pose.pt')
print("✅ Model loaded")

class Pose:
    """Pose keypoints structure"""
    def __init__(self, keypoints):
        # YOLO keypoints: 0=nose, 5=left_shoulder, 6=right_shoulder, 
        # 11=left_hip, 12=right_hip, etc.
        self.left_shoulder = keypoints[5] if len(keypoints) > 5 else [0, 0]
        self.right_shoulder = keypoints[6] if len(keypoints) > 6 else [0, 0]
        self.left_elbow = keypoints[7] if len(keypoints) > 7 else [0, 0]
        self.right_elbow = keypoints[8] if len(keypoints) > 8 else [0, 0]
        self.left_wrist = keypoints[9] if len(keypoints) > 9 else [0, 0]
        self.right_wrist = keypoints[10] if len(keypoints) > 10 else [0, 0]
        self.left_hip = keypoints[11] if len(keypoints) > 11 else [0, 0]
        self.right_hip = keypoints[12] if len(keypoints) > 12 else [0, 0]
        
        # Calculate derived points
        self.neck = [
            (self.left_shoulder[0] + self.right_shoulder[0]) / 2,
            (self.left_shoulder[1] + self.right_shoulder[1]) / 2
        ]
        
        self.shoulder_width = abs(self.right_shoulder[0] - self.left_shoulder[0])
        self.torso_height = abs(self.left_hip[1] - self.left_shoulder[1])

def detect_pose(image):
    """Detect pose using YOLO"""
    results = pose_model(image, verbose=False)
    
    if len(results) == 0 or results[0].keypoints is None:
        return None
    
    keypoints = results[0].keypoints.xy[0].cpu().numpy()
    
    # Check if person is detected
    if len(keypoints) == 0:
        return None
    
    return Pose(keypoints)

def scale_garment(garment, pose, garment_type='top'):
    """Scale garment to fit body using pose"""
    if pose.shoulder_width < 10 or pose.torso_height < 10:
        return garment
    
    # Calculate target dimensions based on garment type
    if garment_type == 'top':
        target_width = int(pose.shoulder_width * 1.3)  # 30% wider than shoulders
        target_height = int(pose.torso_height * 0.9)   # 90% of torso
    elif garment_type == 'dress':
        target_width = int(pose.shoulder_width * 1.3)
        target_height = int(pose.torso_height * 1.8)   # Full length
    elif garment_type == 'bottom':
        hip_width = abs(pose.right_hip[0] - pose.left_hip[0])
        target_width = int(hip_width * 1.2)
        target_height = int(pose.torso_height * 1.2)
    elif garment_type == 'outerwear':
        target_width = int(pose.shoulder_width * 1.5)  # Wider for jackets
        target_height = int(pose.torso_height * 1.1)
    else:
        target_width = int(pose.shoulder_width * 1.2)
        target_height = int(pose.torso_height * 0.9)
    
    # Maintain aspect ratio
    h, w = garment.shape[:2]
    aspect = w / h
    
    if target_width / target_height > aspect:
        # Height-constrained
        new_height = target_height
        new_width = int(new_height * aspect)
    else:
        # Width-constrained
        new_width = target_width
        new_height = int(new_width / aspect)
    
    scaled = cv2.resize(garment, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled

def apply_perspective_warp(garment, pose):
    """Apply perspective transformation to match body angle"""
    # Get shoulder angle
    dx = pose.right_shoulder[0] - pose.left_shoulder[0]
    dy = pose.right_shoulder[1] - pose.left_shoulder[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    
    # Small rotation to match shoulder tilt
    h, w = garment.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid clipping
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    warped = cv2.warpAffine(garment, rotation_matrix, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    
    return warped

def overlay_garment(person_image, garment, pose, garment_type='top'):
    """Overlay garment on person with proper blending"""
    # Scale garment
    scaled_garment = scale_garment(garment, pose, garment_type)
    
    # Apply perspective warp
    warped_garment = apply_perspective_warp(scaled_garment, pose)
    
    h, w = warped_garment.shape[:2]
    
    # Calculate position
    if garment_type == 'top' or garment_type == 'outerwear':
        start_y = int(pose.neck[1] - 10)
        start_x = int(pose.neck[0] - w // 2)
    elif garment_type == 'bottom':
        start_y = int(pose.left_hip[1] - 10)
        start_x = int((pose.left_hip[0] + pose.right_hip[0]) / 2 - w // 2)
    elif garment_type == 'dress':
        start_y = int(pose.neck[1] - 10)
        start_x = int(pose.neck[0] - w // 2)
    else:
        start_y = int(pose.neck[1])
        start_x = int(pose.neck[0] - w // 2)
    
    # Ensure within bounds
    img_h, img_w = person_image.shape[:2]
    
    # Calculate overlay region
    y1 = max(0, start_y)
    y2 = min(img_h, start_y + h)
    x1 = max(0, start_x)
    x2 = min(img_w, start_x + w)
    
    # Calculate garment region
    gy1 = max(0, -start_y)
    gy2 = gy1 + (y2 - y1)
    gx1 = max(0, -start_x)
    gx2 = gx1 + (x2 - x1)
    
    # Check if valid region
    if y2 <= y1 or x2 <= x1 or gy2 <= gy1 or gx2 <= gx1:
        return person_image
    
    # Extract regions
    person_region = person_image[y1:y2, x1:x2]
    garment_region = warped_garment[gy1:gy2, gx1:gx2]
    
    # Alpha blending
    if garment_region.shape[2] == 4:  # Has alpha channel
        alpha = garment_region[:, :, 3:4] / 255.0
        garment_rgb = garment_region[:, :, :3]
        
        # Ensure shapes match
        if person_region.shape[:2] == garment_rgb.shape[:2]:
            blended = (garment_rgb * alpha + person_region * (1 - alpha)).astype(np.uint8)
            person_image[y1:y2, x1:x2] = blended
    else:
        # No alpha channel, use direct overlay
        if person_region.shape[:2] == garment_region.shape[:2]:
            person_image[y1:y2, x1:x2] = garment_region
    
    return person_image

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'ar-tryon-api',
        'model': 'yolov8n-pose'
    }), 200

@app.route('/api/detect-pose', methods=['POST'])
def detect_pose_endpoint():
    """Detect pose keypoints from image"""
    try:
        data = request.get_json()
        
        if not data or 'imageUrl' not in data:
            return jsonify({'error': 'Missing imageUrl'}), 400
        
        # Download image
        response = requests.get(data['imageUrl'], timeout=10)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Detect pose
        pose = detect_pose(img)
        
        if pose is None:
            return jsonify({'detected': False, 'message': 'No person detected'}), 200
        
        return jsonify({
            'detected': True,
            'keypoints': {
                'leftShoulder': {'x': float(pose.left_shoulder[0]), 'y': float(pose.left_shoulder[1])},
                'rightShoulder': {'x': float(pose.right_shoulder[0]), 'y': float(pose.right_shoulder[1])},
                'leftHip': {'x': float(pose.left_hip[0]), 'y': float(pose.left_hip[1])},
                'rightHip': {'x': float(pose.right_hip[0]), 'y': float(pose.right_hip[1])},
                'neck': {'x': float(pose.neck[0]), 'y': float(pose.neck[1])},
            },
            'measurements': {
                'shoulderWidth': float(pose.shoulder_width),
                'torsoHeight': float(pose.torso_height),
            }
        }), 200
        
    except Exception as e:
        print(f"Error in pose detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/try-on', methods=['POST'])
def try_on():
    """Full AR try-on: overlay garment on person"""
    try:
        data = request.get_json()
        
        if not data or 'personImageUrl' not in data or 'garmentImageUrl' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        garment_type = data.get('garmentType', 'top')
        
        # Download person image
        person_response = requests.get(data['personImageUrl'], timeout=10)
        person_array = np.asarray(bytearray(person_response.content), dtype=np.uint8)
        person_img = cv2.imdecode(person_array, cv2.IMREAD_COLOR)
        
        # Download garment image (with transparency)
        garment_response = requests.get(data['garmentImageUrl'], timeout=10)
        garment_array = np.asarray(bytearray(garment_response.content), dtype=np.uint8)
        garment_img = cv2.imdecode(garment_array, cv2.IMREAD_UNCHANGED)
        
        if person_img is None or garment_img is None:
            return jsonify({'error': 'Invalid images'}), 400
        
        # Detect pose
        pose = detect_pose(person_img)
        
        if pose is None:
            return jsonify({'error': 'No person detected in image'}), 400
        
        # Overlay garment
        result_img = overlay_garment(person_img, garment_img, pose, garment_type)
        
        # Encode result as base64
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'resultImage': f'data:image/jpeg;base64,{result_base64}',
            'pose': {
                'shoulderWidth': float(pose.shoulder_width),
                'torsoHeight': float(pose.torso_height),
            }
        }), 200
        
    except Exception as e:
        print(f"Error in try-on: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)