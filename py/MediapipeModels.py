#!/usr/bin/env python
# coding: utf-8

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
import os
import urllib.request

class MediapipeHandModel:
    def __init__(self, model_path='hand_landmarker.task'):
        self.model_path = model_path
        self._download_model_if_needed()
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_model = vision.HandLandmarker.create_from_options(options)
        
        self.mp_drawing = drawing_utils
        self.mp_drawing_styles = drawing_styles
        
        # For backward compatibility with the interface
        class HandsMock:
            HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
        self.mp_hands = HandsMock()
        
    def _download_model_if_needed(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading model to {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
        
    def return_mp_drawing(self):
        return self.mp_drawing
    
    def return_mp_drawing_styles(self):
        return self.mp_drawing_styles
    
    def return_mp_hands(self):
        return self.mp_hands
        
    def return_hand_model(self):
        return self.hand_model
