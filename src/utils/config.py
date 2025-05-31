"""
Configuration Module
Settings and constants used throughout the project
"""

import os
from typing import Dict, Any

# Project information
PROJECT_NAME = "GestureGroove"
VERSION = "1.0.0"
AUTHORS = ["Ümmü Gülsün"]

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Camera settings
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30,
    "device_id": 0
}

# Hand detection settings
HAND_DETECTION_CONFIG = {
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
    "static_image_mode": False
}

# Gesture classification settings
GESTURE_CONFIG = {
    "gestures": ["open_hand", "fist", "point_up", "point_down", "peace"],
    "confidence_threshold": 0.7,
    "smoothing_window": 5
}

# Audio settings
AUDIO_CONFIG = {
    "frequency": 44100,
    "buffer_size": 512,
    "channels": 2,
    "master_volume": 0.7,
    "max_latency_ms": 50
}

# Instrument settings
INSTRUMENTS = {
    "piano": {
        "notes": ["C4", "D4", "E4", "F4", "G4"],
        "base_freq": 261.63  # C4
    },
    "drums": {
        "sounds": ["kick", "snare", "hihat", "crash", "ride"]
    },
    "synth": {
        "wave_types": ["sine", "square", "sawtooth", "triangle"]
    }
}

# UI settings
UI_CONFIG = {
    "window_size": "1200x800",
    "theme": {
        "bg_primary": "#2c3e50",
        "bg_secondary": "#34495e",
        "accent": "#3498db",
        "success": "#2ecc71",
        "danger": "#e74c3c",
        "warning": "#f39c12",
        "text_primary": "#ffffff",
        "text_secondary": "#95a5a6"
    },
    "fonts": {
        "default": ("Arial", 10),
        "heading": ("Arial", 14, "bold"),
        "small": ("Arial", 8)
    }
}

# Beat Challenge settings
BEAT_CONFIG = {
    "default_bpm": 120,
    "patterns": {
        "easy": ["kick", "snare", "kick", "snare"],
        "medium": ["kick", "kick", "snare", "hihat", "kick", "snare"],
        "hard": ["kick", "snare", "hihat", "kick", "hihat", "snare", "kick", "hihat"]
    },
    "scoring": {
        "perfect_hit": 100,
        "good_hit": 50,
        "miss_penalty": -10
    }
}

# Model settings
MODEL_CONFIG = {
    "gesture_model_path": os.path.join(MODELS_DIR, "gesture_model.pkl"),
    "feature_size": 63,  # 21 landmarks * 3 coordinates
    "train_test_split": 0.8
}

# Performance settings
PERFORMANCE_CONFIG = {
    "target_fps": 30,
    "max_processing_time_ms": 33,  # ~30 FPS
    "enable_gpu": False,
    "enable_multithreading": True
}

def get_config() -> Dict[str, Any]:
    """Returns all configuration"""
    return {
        "project": {
            "name": PROJECT_NAME,
            "version": VERSION,
            "authors": AUTHORS
        },
        "paths": {
            "base": BASE_DIR,
            "assets": ASSETS_DIR,
            "sounds": SOUNDS_DIR,
            "models": MODELS_DIR
        },
        "camera": CAMERA_CONFIG,
        "hand_detection": HAND_DETECTION_CONFIG,
        "gesture": GESTURE_CONFIG,
        "audio": AUDIO_CONFIG,
        "instruments": INSTRUMENTS,
        "ui": UI_CONFIG,
        "beat": BEAT_CONFIG,
        "model": MODEL_CONFIG,
        "performance": PERFORMANCE_CONFIG
    }

def create_directories():
    """Creates necessary directories"""
    dirs_to_create = [
        ASSETS_DIR,
        SOUNDS_DIR,
        MODELS_DIR,
        os.path.join(SOUNDS_DIR, "drums"),
        os.path.join(SOUNDS_DIR, "piano"),
        os.path.join(SOUNDS_DIR, "synth"),
        os.path.join(ASSETS_DIR, "images", "icons")
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        
    print(f"✅ Project directories created: {BASE_DIR}")

if __name__ == "__main__":
    # Test
    create_directories()
    config = get_config()
    print("✅ Configuration loaded successfully!")
    print(f"📁 Base directory: {config['paths']['base']}")
    print(f"🎵 Audio frequency: {config['audio']['frequency']}Hz")
    print(f"📷 Camera resolution: {config['camera']['width']}x{config['camera']['height']}") 