from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import base64
import json
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ASLRecognizer:
    def __init__(self, model_path='models/best_model2.keras'):
        self.model_path = model_path
        self.model = None
        self.actions = ['welcome', 'we', 'happy', 'you', 'here', 'today', 'topic', 'c', 't', 'i', 's', 'a', 'l']
        self.seq_length = 15
        self.seq = []
        self.action_seq = []
        self.lock = Lock()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logger.info(f"✅ Model loaded successfully from {self.model_path}")
                logger.info(f"Model output shape: {self.model.output_shape}")
                
                # Verify actions match model output
                if len(self.actions) != self.model.output_shape[-1]:
                    logger.warning(f"Actions count ({len(self.actions)}) doesn't match model output ({self.model.output_shape[-1]})")
            else:
                logger.error(f"❌ Model file not found: {self.model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def extract_features(self, hand_landmarks):
        """Extract features from hand landmarks (from your dynamic_test.py)"""
        joint = np.zeros((21, 4))
        for j, lm in enumerate(hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # Compute angles between joints (exact logic from your code)
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
        v = v2 - v1
        
        # Normalize vectors
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Calculate angles
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
        angle = np.degrees(angle)

        # Create feature array (same as your dynamic_test.py)
        d = np.concatenate([joint.flatten(), angle])
        return d
    
    def process_frame(self, frame):
        """Process a single frame and return predictions"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        with self.lock:
            try:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hands
                results = self.hands.process(frame_rgb)
                
                predictions = []
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract features using your method
                        features = self.extract_features(hand_landmarks)
                        self.seq.append(features)
                        
                        # Make prediction when we have enough frames
                        if len(self.seq) >= self.seq_length:
                            # Prepare input data
                            input_data = np.expand_dims(
                                np.array(self.seq[-self.seq_length:], dtype=np.float32), 
                                axis=0
                            )
                            
                            # Get prediction
                            y_pred = self.model.predict(input_data, verbose=0)[0]
                            
                            # Get top predictions
                            top_indices = np.argsort(y_pred)[::-1][:3]
                            
                            for idx in top_indices:
                                confidence = float(y_pred[idx])
                                if confidence > 0.1:  # Only show predictions with some confidence
                                    predictions.append({
                                        'action': self.actions[idx],
                                        'confidence': confidence
                                    })
                        
                        # Keep sequences from getting too long
                        if len(self.seq) > self.seq_length:
                            self.seq = self.seq[-self.seq_length:]
                
                return {"predictions": predictions, "status": "success"}
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                return {"error": str(e)}

# Initialize recognizer
recognizer = ASLRecognizer()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Check if model is loaded and ready"""
    return jsonify({
        "model_loaded": recognizer.model is not None,
        "actions": recognizer.actions,
        "model_path": recognizer.model_path
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Process frame and return predictions"""
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"})
        
        # Process frame
        result = recognizer.process_frame(frame)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/reset')
def reset():
    """Reset the sequence buffer"""
    with recognizer.lock:
        recognizer.seq = []
        recognizer.action_seq = []
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    print("🚀 Starting ASL Recognition Server...")
    print(f"📁 Looking for model at: {recognizer.model_path}")
    print(f"🎯 Configured actions: {recognizer.actions}")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)