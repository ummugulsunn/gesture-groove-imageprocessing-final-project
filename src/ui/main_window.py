"""
Main Window Module - Sueda's Task
Main user interface using Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Scale, Canvas, LabelFrame
import cv2
from PIL import Image, ImageTk
import threading
import time
from typing import Optional, Callable, List, Tuple
import numpy as np


class MainWindow:
    """
    Main application window
    
    Features to be developed by Sueda:
    - Modern and user-friendly interface design
    - Real-time camera preview
    - Control panel (volume, instrument selection)
    - Score and statistics display
    - Error handling and user feedback
    """
    
    def __init__(self):
        """Initializes MainWindow"""
        self.root = tk.Tk()
        self.root.title("🎵 GestureGroove - Hand Gesture Music Player")
        self.root.geometry("1400x900")  # Made a bit larger
        self.root.configure(bg='#2c3e50')
        
        # Set window icon (if exists)
        try:
            # No icon but we can use emoji in title
            pass
        except:
            pass
        
        # Application state
        self.is_camera_running = False
        self.current_gesture = None
        self.score = 0
        self.fps = 0
        
        # Gesture thresholds (adjustable)
        self.gesture_confidence_threshold = 0.6
        self.gesture_stability_frames = 3  # Same gesture for N frames
        self.gesture_frame_counter = {}
        
        # Callback functions
        self.gesture_callback: Optional[Callable] = None
        self.camera_callback: Optional[Callable] = None
        
        # UI components
        self.camera_frame = None
        self.control_panel = None
        self.status_bar = None
        self.visualizer_frame = None
        
        self.beat_mode = False
        self.score = 0  # Beat challenge score
        
        # Default theme (setup_styles will update this later)
        self.theme = {
            'bg_color': '#2c3e50',      # Default background
            'text_color': '#ecf0f1',    # Default text color
            'accent_color': '#3498db',  # Accent color
            'accent_fg_color': 'white',
            'secondary_bg_color': '#34495e',
            'font_family': 'Arial'      # Default font
        }
        
        # Beat Challenge UI Elements
        self.beat_challenge_feedback_label: Optional[ttk.Label] = None
        self.beat_challenge_score_label: Optional[ttk.Label] = None
        self.beat_challenge_combo_label: Optional[ttk.Label] = None  # Combo indicator
        self.beat_challenge_guide_label: Optional[ttk.Label] = None  # Guide label
        self.pattern_display_labels: List[ttk.Label] = []  # For displaying pattern steps
        
        # Emoji map for pattern display
        self.emoji_map = {
            "kick": "🥁",
            "snare": " snare ",
            "hihat": "🎩",
            "crash": "💥",
            "tom": "🛢️",
            "0": "➖",
            "default": "❓",
            # Additions for gestures:
            "fist": "✊",
            "point_up": "☝️",
            "peace": "✌️",
            "point_down": "👇",
            "open_hand": "🖐️"
        }
        
        self.feedback_colors = {
            "perfect": ("#4CAF50", "white"),  # Green
            "good": ("#8BC34A", "black"),     # Light Green
            "miss": ("#F44336", "white"),     # Red
            "wrong_sound": ("#FF9800", "black"),  # Orange
            "early_late": ("#FFC107", "black"),  # Amber
            "default": (self.theme['bg_color'], self.theme['text_color'])
        }
        
        self.setup_styles()
        self.setup_ui()
        self._setup_beat_challenge_ui()
        
        print("✅ MainWindow initialized")
        
    def setup_styles(self):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Modern UI theme and styles
        """
        # Use ttk.Style for custom style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors
        style.configure('Custom.TButton',
                       background='#3498db',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Custom.TButton',
                 background=[('active', '#2980b9')])
        
        # Beat Challenge Active style
        style.configure('Challenge.TButton',
                       background='#e74c3c',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Challenge.TButton',
                 background=[('active', '#c0392b')])
        
        # General Label Style
        style.configure('TLabel', 
                        background=self.theme['bg_color'], 
                        foreground=self.theme['text_color'], 
                        font=(self.theme['font_family'], 10))

        # Accent Label Style
        style.configure('Accent.TLabel', 
                        background=self.theme.get('accent_label_bg', self.theme['bg_color']),  # Custom accent bg if theme supports
                        foreground=self.theme['accent_color'], 
                        font=(self.theme['font_family'], 12, 'bold'))

        # Secondary Label Style (e.g., for pattern display)
        style.configure('Secondary.TLabel', 
                        background=self.theme['secondary_bg_color'], 
                        foreground=self.theme.get('secondary_text_color', self.theme['text_color']), 
                        font=(self.theme['font_family'], 10))

        # General LabelFrame Style (Simplified)
        style.configure('TLabelFrame', 
                        background=self.theme['secondary_bg_color'])
        style.configure('TLabelFrame.Label', 
                        foreground=self.theme['text_color'], 
                        background=self.theme['secondary_bg_color'],  # Should match LabelFrame's own bg
                        font=(self.theme['font_family'], 11, 'bold'))

        # General Frame style (TFrame for pattern display)
        style.configure('TFrame', background=self.theme['secondary_bg_color'])
        
        # TODO: Sueda will add more styles
        
    def setup_ui(self):
        """Creates main UI layout"""
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel - Camera and controls
        top_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, bd=2)
        top_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.setup_camera_frame(top_panel)
        self.setup_control_panel(top_panel)
        
        # Bottom panel - Visualization and statistics
        bottom_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, bd=2)
        bottom_panel.pack(fill=tk.X, pady=(0, 10))
        
        self.setup_visualizer_frame(bottom_panel)
        
        # Status bar
        self.setup_status_bar(main_container)
        
    def setup_camera_frame(self, parent):
        """Creates camera preview area"""
        # Main camera container
        camera_main_container = tk.Frame(parent, bg='#34495e')
        camera_main_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top title
        title_label = tk.Label(camera_main_container, 
                             text="📷 Real-Time Gesture Detection",
                             font=('Arial', 16, 'bold'),
                             bg='#34495e', fg='white')
        title_label.pack(pady=(0, 10))
        
        # Bottom container - gesture display + camera
        content_container = tk.Frame(camera_main_container, bg='#34495e')
        content_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Gesture Display
        self.setup_gesture_display_panel(content_container)
        
        # Right panel - Camera
        camera_container = tk.Frame(content_container, bg='#34495e')
        camera_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Camera title
        cam_title = tk.Label(camera_container,
                           text="🎥 Camera Feed",
                           font=('Arial', 12, 'bold'),
                           bg='#34495e', fg='white')
        cam_title.pack(pady=(0, 5))
        
        # Camera display area
        self.camera_frame = tk.Label(camera_container,
                                   text="Camera starting...",
                                   bg='black', fg='white',
                                   width=50, height=25,
                                   font=('Arial', 10))
        self.camera_frame.pack(expand=True, fill=tk.BOTH)
        
    def setup_gesture_display_panel(self, parent):
        """Left side large gesture display panel"""
        # Gesture display container
        gesture_container = tk.Frame(parent, bg='#2c3e50', relief=tk.RAISED, bd=2)
        gesture_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Panel title
        gesture_title = tk.Label(gesture_container,
                               text="🖐️ Detected Gesture",
                               font=('Arial', 14, 'bold'),
                               bg='#2c3e50', fg='white')
        gesture_title.pack(pady=(10, 5))
        
        # Main gesture display - BIG
        self.main_gesture_display = tk.Label(gesture_container,
                                           text="❓",
                                           font=('Arial', 48, 'bold'),
                                           bg='#34495e', fg='#ecf0f1',
                                           width=8, height=3,
                                           relief=tk.RAISED, bd=3)
        self.main_gesture_display.pack(pady=10, padx=20)
        
        # Gesture name
        self.gesture_name_label = tk.Label(gesture_container,
                                         text="No Gesture",
                                         font=('Arial', 14, 'bold'),
                                         bg='#2c3e50', fg='#95a5a6')
        self.gesture_name_label.pack(pady=(0, 5))
        
        # Current instrument display
        self.current_instrument_label = tk.Label(gesture_container,
                                                text="🎹 Piano",
                                                font=('Arial', 12),
                                                bg='#2c3e50', fg='#3498db')
        self.current_instrument_label.pack(pady=(0, 10))
        
        # Confidence indicator
        confidence_frame = tk.Frame(gesture_container, bg='#2c3e50')
        confidence_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        tk.Label(confidence_frame,
                text="Confidence:",
                font=('Arial', 10),
                bg='#2c3e50', fg='white').pack()
        
        self.confidence_bar = tk.Canvas(confidence_frame,
                                      height=20, bg='#34495e',
                                      highlightthickness=0)
        self.confidence_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Hand landmarks visualization
        landmarks_frame = tk.LabelFrame(gesture_container,
                                      text="Hand Landmarks",
                                      font=('Arial', 10, 'bold'),
                                      background='#2c3e50', fg='white')
        landmarks_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
        
        self.landmarks_canvas = tk.Canvas(landmarks_frame,
                                        width=150, height=150,
                                        bg='black',
                                        highlightthickness=0)
        self.landmarks_canvas.pack(pady=10)
        
        # Gesture history
        history_frame = tk.LabelFrame(gesture_container,
                                    text="Recent Gestures",
                                    font=('Arial', 10, 'bold'),
                                    background='#2c3e50', fg='white')
        history_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        self.gesture_history_frame = tk.Frame(history_frame, bg='#2c3e50')
        self.gesture_history_frame.pack(pady=5)
        
        # Initially empty history
        self.gesture_history = []
        self._update_gesture_history_display()
        
    def setup_control_panel(self, parent):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Control panel design
        """
        # Control panel
        control_container = tk.Frame(parent, bg='#34495e')
        control_container.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.right_panel = control_container
        
        # Title
        control_title = tk.Label(control_container,
                               text="🎛️ Control Panel",
                               font=('Arial', 14, 'bold'),
                               bg='#34495e', fg='white')
        control_title.pack(pady=(0, 20))
        
        # Camera control
        camera_section = tk.LabelFrame(control_container,
                                     text="Camera",
                                     font=('Arial', 10, 'bold'),
                                     bg='#34495e', fg='white')
        camera_section.pack(fill=tk.X, pady=(0, 10))
        
        self.camera_button = ttk.Button(camera_section,
                                      text="📷 Start Camera",
                                      style='Custom.TButton',
                                      command=self.toggle_camera)
        self.camera_button.pack(pady=10, padx=10, fill=tk.X)
        
        # Audio control
        audio_section = tk.LabelFrame(control_container,
                                    text="Audio Settings",
                                    font=('Arial', 10, 'bold'),
                                    bg='#34495e', fg='white')
        audio_section.pack(fill=tk.X, pady=(0, 10))
        
        # Volume slider
        tk.Label(audio_section, text="🔊 Volume Level:",
                bg='#34495e', fg='white').pack(anchor=tk.W, padx=10)
        
        self.volume_var = tk.DoubleVar(value=70)
        self.volume_scale = ttk.Scale(audio_section,
                                    from_=0, to=100,
                                    variable=self.volume_var,
                                    orient=tk.HORIZONTAL,
                                    command=self.on_volume_change)
        self.volume_scale.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Instrument selection
        tk.Label(audio_section, text="🎹 Instrument:",
                bg='#34495e', fg='white').pack(anchor=tk.W, padx=10)
        
        self.instrument_var = tk.StringVar(value="Piano")
        self.instrument_combo = ttk.Combobox(audio_section,
                                           textvariable=self.instrument_var,
                                           values=["Piano", "Drums", "Synth", "Guitar"],
                                           state="readonly")
        self.instrument_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Instrument change event bind et
        self.instrument_combo.bind('<<ComboboxSelected>>', self.on_instrument_change_event)
        
        # Beat Challenge
        beat_section = tk.LabelFrame(control_container,
                                   text="Beat Challenge",
                                   font=('Arial', 10, 'bold'),
                                   bg='#34495e', fg='white')
        beat_section.pack(fill=tk.X, pady=(0, 10))
        
        self.beat_challenge_button = ttk.Button(beat_section,
                                    text="🎯 Start Challenge",
                                    style='Custom.TButton',
                                    command=self.start_beat_challenge)
        self.beat_challenge_button.pack(pady=10, padx=10, fill=tk.X)
        
        # Score display
        self.score_label = tk.Label(beat_section,
                                  text="Score: 0",
                                  font=('Arial', 12, 'bold'),
                                  bg='#34495e', fg='#e74c3c')
        self.score_label.pack(pady=(0, 10))
        
        # Settings Section
        settings_section = tk.LabelFrame(control_container,
                                       text="⚙️ Settings",
                                       font=('Arial', 10, 'bold'),
                                       bg='#34495e', fg='white')
        settings_section.pack(fill=tk.X, pady=(0, 10))
        
        # Gesture Sensitivity
        tk.Label(settings_section, text="🎯 Gesture Sensitivity:",
                bg='#34495e', fg='white', font=('Arial', 9)).pack(anchor=tk.W, padx=10)
        
        self.sensitivity_var = tk.DoubleVar(value=0.6)
        self.sensitivity_scale = ttk.Scale(settings_section,
                                         from_=0.3, to=0.9,
                                         variable=self.sensitivity_var,
                                         orient=tk.HORIZONTAL,
                                         command=self.on_sensitivity_change)
        self.sensitivity_scale.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Stability Frames
        tk.Label(settings_section, text="🔒 Stability (frames):",
                bg='#34495e', fg='white', font=('Arial', 9)).pack(anchor=tk.W, padx=10)
        
        self.stability_var = tk.IntVar(value=3)
        stability_frame = tk.Frame(settings_section, bg='#34495e')
        stability_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        for val in [1, 3, 5]:
            rb = tk.Radiobutton(stability_frame,
                              text=str(val),
                              variable=self.stability_var,
                              value=val,
                              bg='#34495e', fg='white',
                              selectcolor='#2ecc71',
                              command=self.on_stability_change)
            rb.pack(side=tk.LEFT, padx=5)
        
    def setup_visualizer_frame(self, parent):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Visualization panel
        """
        # Visualization container
        viz_container = tk.Frame(parent, bg='#34495e')
        viz_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        viz_title = tk.Label(viz_container,
                           text="📊 Gesture Visualization",
                           font=('Arial', 14, 'bold'),
                           bg='#34495e', fg='white')
        viz_title.pack(pady=(0, 10))
        
        # Gesture indicators
        gestures = ['✊ Fist', '✋ Open Hand', '☝️ Point Up', '👇 Point Down', '✌️ Peace']
        self.gesture_indicators = {}
        
        for i, gesture in enumerate(gestures):
            indicator = tk.Label(viz_container,
                               text=gesture,
                               font=('Arial', 12),
                               bg='#7f8c8d', fg='white',
                               relief=tk.RAISED, bd=2,
                               padx=20, pady=10)
            indicator.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.gesture_indicators[gesture.split()[1].lower()] = indicator
            
    def setup_status_bar(self, parent):
        """Creates status bar"""
        self.status_bar = tk.Frame(parent, bg='#34495e', relief=tk.SUNKEN, bd=1)
        self.status_bar.pack(fill=tk.X)
        
        # FPS indicator
        self.fps_label = tk.Label(self.status_bar,
                                text="FPS: 0",
                                bg='#34495e', fg='#2ecc71',
                                font=('Arial', 10))
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        # Status message
        self.status_label = tk.Label(self.status_bar,
                                   text="Ready",
                                   bg='#34495e', fg='white',
                                   font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Time
        self.time_label = tk.Label(self.status_bar,
                                 text="",
                                 bg='#34495e', fg='#95a5a6',
                                 font=('Arial', 10))
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        self.update_time()
        
    def toggle_camera(self):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Camera on/off functionality
        
        Implementation Features:
        1. Safe camera start/stop
        2. UI state management
        3. Error handling
        4. User feedback
        """
        try:
            if not self.is_camera_running:
                self.start_camera()
            else:
                self.stop_camera()
        except Exception as e:
            self.show_error_message("Camera Error", f"Camera operation failed: {e}")
            
    def start_camera(self):
        """Starts the camera"""
        try:
            self.is_camera_running = True
            self.camera_button.configure(text="📷 Stop Camera")
            self.update_status("Camera starting...")
            
            # UI state update
            self.camera_frame.configure(bg='#1a1a1a', text="Camera turning on...")
            
            # Start camera thread
            if self.camera_callback:
                camera_thread = threading.Thread(target=self.camera_callback, daemon=True)
                camera_thread.start()
                self.update_status("Camera active")
            else:
                self.update_status("Camera callback not found")
                
        except Exception as e:
            self.is_camera_running = False
            self.camera_button.configure(text="📷 Start Camera")
            self.show_error_message("Camera Start Error", str(e))
            
    def stop_camera(self):
        """Stops the camera"""
        try:
            self.is_camera_running = False
            self.camera_button.configure(text="📷 Start Camera")
            self.update_status("Camera stopped")
            
            # Clear camera frame
            self.camera_frame.configure(image='', text="Camera stopped", bg='black')
            if hasattr(self.camera_frame, 'image'):
                self.camera_frame.image = None
                
        except Exception as e:
            self.show_error_message("Camera Stop Error", str(e))
        
    def update_camera_frame(self, cv_image):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Updates the camera frame
        
        Optimizations to implement:
        1. Efficient image resizing
        2. Memory management
        3. Frame rate limiting
        4. Error recovery
        """
        if not self.is_camera_running:
            return
            
        try:
            if cv_image is None:
                return
                
            # Convert OpenCV BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Resize for Tkinter (maintain aspect ratio)
            frame_width = self.camera_frame.winfo_width()
            frame_height = self.camera_frame.winfo_height()
            
            if frame_width > 1 and frame_height > 1:
                # Calculate aspect ratio
                img_width, img_height = pil_image.size
                aspect_ratio = img_width / img_height
                
                # Calculate new dimensions
                if frame_width / frame_height > aspect_ratio:
                    new_height = frame_height - 20
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = frame_width - 20
                    new_height = int(new_width / aspect_ratio)
                
                display_size = (max(new_width, 1), max(new_height, 1))
            else:
                display_size = (640, 480)  # Default size
            
            # High quality resize
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update frame (main thread)
            self.root.after(0, lambda: self._update_camera_display(photo))
            
        except Exception as e:
            print(f"❌ Camera update error: {e}")
            self.root.after(0, lambda: self.update_status(f"Camera error: {str(e)[:30]}..."))
    
    def _update_camera_display(self, photo):
        """Camera display update helper (main thread)"""
        try:
            self.camera_frame.configure(image=photo, text="")
            self.camera_frame.image = photo  # Reference hold
        except Exception as e:
            print(f"❌ Display update error: {e}")
            
    def update_gesture_display(self, gesture: str, confidence: float = 0.0):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Displays the current gesture visually
        
        Animation features to implement:
        1. Smooth color transitions
        2. Pulse effects for active gestures
        3. Gesture history display
        4. Confidence level indicators
        """
        try:
            # Gesture stability check
            stable_gesture = self._check_gesture_stability(gesture, confidence)
            
            # Hand detection status update
            if stable_gesture and stable_gesture != 'none':
                detection_status = f"🖐️ Hand Detected: {stable_gesture.upper()}"
                if confidence > 0:
                    detection_status += f" ({confidence:.1%})"
                self.update_status(detection_status, 'success')
            else:
                self.update_status("🔍 Searching for hands...", 'info')
            
            # Main gesture display update
            self._update_main_gesture_display(stable_gesture, confidence)
            
            # Gesture mapping (old system compatibility)
            gesture_map = {
                'open_hand': 'open',
                'fist': 'fist', 
                'point_up': 'up',
                'point_down': 'down',
                'peace': 'peace'
            }
            
            # Deactivate all indicators (bottom panel)
            for indicator in self.gesture_indicators.values():
                indicator.configure(bg='#7f8c8d', relief=tk.RAISED)
                
            # Activate highlighted gesture (bottom panel)
            if stable_gesture and stable_gesture != 'none':
                display_gesture = gesture_map.get(stable_gesture, stable_gesture)
                if display_gesture and display_gesture in self.gesture_indicators:
                    active_indicator = self.gesture_indicators[display_gesture]
                    active_indicator.configure(bg='#e74c3c', relief=tk.SUNKEN)
                    
                    # Start pulse effect
                    self._start_pulse_effect(active_indicator)
                    
                self.current_gesture = stable_gesture
                
                # Gesture history update
                self._add_to_gesture_history(stable_gesture)
                        
        except Exception as e:
            print(f"❌ Gesture display update error: {e}")
    
    def _check_gesture_stability(self, gesture: str, confidence: float) -> str:
        """Gesture stability check - same gesture must be detected N frames in a row"""
        try:
            # Confidence threshold check
            if confidence < self.gesture_confidence_threshold:
                return None
            
            # Gesture not found
            if not gesture or gesture == 'none':
                # Counters reset
                self.gesture_frame_counter = {}
                return None
            
            # Frame counter update
            if gesture in self.gesture_frame_counter:
                self.gesture_frame_counter[gesture] += 1
            else:
                # New gesture - reset others
                self.gesture_frame_counter = {gesture: 1}
            
            # Stability check
            if self.gesture_frame_counter[gesture] >= self.gesture_stability_frames:
                return gesture
            else:
                # Still not stable enough
                return self.current_gesture  # Return previous stable gesture
                
        except Exception as e:
            print(f"❌ Gesture stability check error: {e}")
            return gesture  # Fallback
    
    def _update_main_gesture_display(self, gesture: str, confidence: float = 0.0):
        """Updates main gesture display"""
        try:
            # Gesture emojis
            gesture_emojis = {
                'open_hand': '✋',
                'fist': '✊',
                'point_up': '☝️',
                'point_down': '👇',
                'peace': '✌️'
            }
            
            # Gesture names
            gesture_names = {
                'open_hand': 'Open Hand',
                'fist': 'Fist',
                'point_up': 'Point Up',
                'point_down': 'Point Down',
                'peace': 'Peace Sign'
            }
            
            if gesture and gesture != 'none':
                # Emoji and name update
                emoji = gesture_emojis.get(gesture, '🖐️')
                name = gesture_names.get(gesture, gesture.title())
                
                self.main_gesture_display.configure(
                    text=emoji,
                    bg='#2ecc71',  # Green - active
                    fg='white'
                )
                
                self.gesture_name_label.configure(
                    text=name,
                    fg='#2ecc71'
                )
                
                # Start pulse animation
                self._start_main_gesture_pulse()
                
            else:
                # Gesture not found
                self.main_gesture_display.configure(
                    text='❓',
                    bg='#34495e',  # Gray - passive
                    fg='#95a5a6'
                )
                
                self.gesture_name_label.configure(
                    text='No Gesture',
                    fg='#95a5a6'
                )
            
            # Confidence bar update
            self._update_confidence_bar(confidence)
            
        except Exception as e:
            print(f"❌ Main gesture display error: {e}")
    
    def _start_main_gesture_pulse(self):
        """Pulse effect for main gesture display"""
        def pulse(step=0):
            if step < 4:  # 2 pulse cycle
                colors = ['#2ecc71', '#27ae60'] if step % 2 == 0 else ['#27ae60', '#2ecc71']
                try:
                    self.main_gesture_display.configure(bg=colors[0])
                    self.root.after(300, lambda: pulse(step + 1))
                except:
                    pass  # Widget destroyed
                    
        pulse()
    
    def _update_confidence_bar(self, confidence: float):
        """Updates confidence progress bar"""
        try:
            self.confidence_bar.delete("all")
            
            # Bar dimensions
            bar_width = self.confidence_bar.winfo_width()
            bar_height = self.confidence_bar.winfo_height()
            
            if bar_width <= 1:  # Canvas not yet rendered
                self.root.after(100, lambda: self._update_confidence_bar(confidence))
                return
            
            # Progress bar fill
            fill_width = int(bar_width * confidence)
            
            # Background
            self.confidence_bar.create_rectangle(
                0, 0, bar_width, bar_height,
                fill='#34495e', outline='#2c3e50'
            )
            
            # Progress fill
            if fill_width > 0:
                color = '#e74c3c' if confidence < 0.3 else '#f39c12' if confidence < 0.7 else '#2ecc71'
                self.confidence_bar.create_rectangle(
                    0, 0, fill_width, bar_height,
                    fill=color, outline=''
                )
            
            # Text
            self.confidence_bar.create_text(
                bar_width//2, bar_height//2,
                text=f"{confidence:.1%}",
                fill='white', font=('Arial', 8, 'bold')
            )
            
        except Exception as e:
            print(f"❌ Confidence bar error: {e}")
    
    def _add_to_gesture_history(self, gesture: str):
        """Add to gesture history"""
        try:
            if gesture and gesture != 'none':
                # Add to history
                self.gesture_history.append(gesture)
                
                # Keep last 5 gestures
                if len(self.gesture_history) > 5:
                    self.gesture_history.pop(0)
                
                # Display update
                self._update_gesture_history_display()
                
        except Exception as e:
            print(f"❌ Gesture history error: {e}")
    
    def _update_gesture_history_display(self):
        """Updates gesture history display"""
        try:
            # Clear old widgets
            for widget in self.gesture_history_frame.winfo_children():
                widget.destroy()
            
            # Gesture emojis
            gesture_emojis = {
                'open_hand': '✋',
                'fist': '✊',
                'point_up': '☝️',
                'point_down': '👇',
                'peace': '✌️'
            }
            
            if not self.gesture_history:
                # Empty history
                empty_label = tk.Label(self.gesture_history_frame,
                                     text="No recent gestures",
                                     font=('Arial', 8),
                                     bg='#2c3e50', fg='#95a5a6')
                empty_label.pack()
            else:
                # Show last gestures (newest first)
                for i, gesture in enumerate(reversed(self.gesture_history[-5:])):
                    emoji = gesture_emojis.get(gesture, '🖐️')
                    
                    alpha = 1.0 - (i * 0.15)  # Older gestures lighter
                    
                    gesture_widget = tk.Label(self.gesture_history_frame,
                                            text=emoji,
                                            font=('Arial', 16),
                                            bg='#2c3e50',
                                            fg='white' if alpha > 0.7 else '#95a5a6')
                    gesture_widget.pack(side=tk.LEFT, padx=2)
                    
        except Exception as e:
            print(f"❌ Gesture history display error: {e}")
    
    def update_hand_landmarks(self, landmarks):
        """Updates hand landmarks visualization"""
        try:
            if not hasattr(self, 'landmarks_canvas'):
                return
                
            self.landmarks_canvas.delete("all")
            
            if landmarks is None:
                # No hand detected
                self.landmarks_canvas.create_text(
                    75, 75, text="No Hand\nDetected",
                    fill='#95a5a6', font=('Arial', 10), justify=tk.CENTER
                )
                return
            
            # Landmarks'ı çiz
            canvas_width = 150
            canvas_height = 150
            
            # Convert to numpy array
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)
            
            # Reshape if needed
            if landmarks.ndim == 1 and len(landmarks) >= 42:
                if len(landmarks) >= 63:
                    landmarks = landmarks[:63].reshape(21, 3)
                else:
                    landmarks = landmarks[:42].reshape(21, 2)
            
            if len(landmarks) >= 21:
                # Normalize coordinates
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                
                # Convert to canvas coordinates
                margin = 10
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                if x_max > x_min and y_max > y_min:
                    scale_x = (canvas_width - 2*margin) / (x_max - x_min)
                    scale_y = (canvas_height - 2*margin) / (y_max - y_min)
                    scale = min(scale_x, scale_y)
                    
                    canvas_x = margin + (x_coords - x_min) * scale
                    canvas_y = margin + (y_coords - y_min) * scale
                    
                    # Draw landmarks
                    for i in range(len(canvas_x)):
                        x, y = canvas_x[i], canvas_y[i]
                        
                        # Different colors for finger tips
                        if i in [4, 8, 12, 16, 20]:  # Finger tips
                            color = '#e74c3c'
                            size = 4
                        else:
                            color = '#3498db'
                            size = 2
                            
                        self.landmarks_canvas.create_oval(
                            x-size, y-size, x+size, y+size,
                            fill=color, outline='white'
                        )
                        
                        # Landmark number (selected ones)
                        if i in [0, 4, 8, 12, 16, 20]:
                            self.landmarks_canvas.create_text(
                                x+8, y, text=str(i),
                                fill='white', font=('Arial', 6)
                            )
            
        except Exception as e:
            print(f"❌ Hand landmarks visualization error: {e}")
    
    def start_beat_challenge(self):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Starts the Beat Challenge mode
        
        Features to implement:
        1. Pattern visualization
        2. Score tracking with animations
        3. Difficulty progression
        4. Visual metronome
        5. Success/failure feedback
        """
        try:
            # Call main application beat challenge callback
            if hasattr(self, 'beat_challenge_callback') and self.beat_challenge_callback:
                self.beat_challenge_callback()
            
            self.score = 0
            self.update_score(0)
            self.beat_challenge_button.configure(text="🎯 Active Challenge", style='Challenge.TButton')
            self.update_status("Beat Challenge started!")
            
            # Create challenge UI
            self._create_beat_challenge_ui()
            
            # Default pattern
            default_pattern = ['kick', '0', 'snare', '0', 'kick', 'kick', 'snare', '0']
            self._display_beat_pattern(default_pattern)
            
            # Challenge timer start
            self.challenge_time = 30  # 30 seconds
            self._start_challenge_timer()
            
        except Exception as e:
            self.show_error_message("Challenge Error", f"Beat Challenge failed to start: {e}")
    
    def _create_beat_challenge_ui(self):
        """Creates additional UI for Beat Challenge"""
        if hasattr(self, 'challenge_frame'):
            self.challenge_frame.destroy()
            
        # Challenge info frame
        self.challenge_frame = tk.Frame(self.visualizer_frame, bg='#34495e')
        self.challenge_frame.pack(fill=tk.X, pady=5)
        
        # Timer
        self.timer_label = tk.Label(self.challenge_frame,
                                  text="⏱️ 30",
                                  font=('Arial', 16, 'bold'),
                                  bg='#34495e', fg='#f39c12')
        self.timer_label.pack(side=tk.LEFT, padx=10)
        
        # Pattern display
        self.pattern_frame = tk.Frame(self.challenge_frame, bg='#34495e')
        self.pattern_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(self.pattern_frame,
                text="🥁 Pattern:",
                font=('Arial', 12, 'bold'),
                bg='#34495e', fg='white').pack(side=tk.LEFT)
                
        # Target display
        self.target_label = tk.Label(self.challenge_frame,
                                   text="🎯 Target: ---",
                                   font=('Arial', 12, 'bold'),
                                   bg='#34495e', fg='#2ecc71')
        self.target_label.pack(side=tk.RIGHT, padx=10)
    
    def _display_beat_pattern(self, pattern):
        """Displays the beat pattern visually"""
        if not hasattr(self, 'pattern_indicators'):
            self.pattern_indicators = []
        # Clear old indicators
        for indicator in self.pattern_indicators:
            indicator.destroy()
        self.pattern_indicators = []
        # Create new pattern indicators
        for i, step in enumerate(pattern):
            # Select emoji for gesture or drum step
            display_text = self.emoji_map.get(step, self.emoji_map["default"])
            color = '#e74c3c' if step != '0' else '#7f8c8d'
            indicator = tk.Label(self.pattern_frame,
                               text=display_text,
                               width=3, height=1,
                               bg=color, fg='white',
                               font=('Arial', 16, 'bold'),
                               relief=tk.RAISED, bd=1)
            indicator.pack(side=tk.LEFT, padx=1)
            self.pattern_indicators.append(indicator)
    
    def _start_challenge_timer(self):
        """Starts the challenge timer"""
        def countdown():
            if hasattr(self, 'challenge_time') and self.challenge_time > 0:
                self.timer_label.configure(text=f"⏱️ {self.challenge_time}")
                self.challenge_time -= 1
                self.root.after(1000, countdown)
            else:
                self._end_challenge()
                
        countdown()
    
    def _end_challenge(self):
        """Ends the challenge"""
        self.beat_challenge_button.configure(text="🎯 Start Challenge", style='Custom.TButton')
        self.update_status(f"Challenge ended! Final Score: {self.score}")
        
        if hasattr(self, 'challenge_frame'):
            self.challenge_frame.destroy()
            
        # Success message
        if self.score > 50:
            self.show_success_message("Congratulations!", f"Great performance! Score: {self.score}")
        else:
            self.show_info_message("Challenge Ended", f"Score: {self.score}. Try again!")
    
    def update_fps(self, fps: float):
        """Updates FPS indicator"""
        try:
            color = '#2ecc71' if fps > 20 else '#f39c12' if fps > 10 else '#e74c3c'
            self.fps_label.configure(text=f"FPS: {fps:.1f}", fg=color)
        except Exception as e:
            print(f"❌ FPS update error: {e}")
    
    def update_status(self, message: str, message_type: str = "info"):
        """Updates status bar message"""
        try:
            colors = {
                'info': '#ffffff',
                'success': '#2ecc71', 
                'warning': '#f39c12',
                'error': '#e74c3c'
            }
            color = colors.get(message_type, '#ffffff')
            self.status_label.configure(text=message[:50], fg=color)
            
            # Auto-clear after 5 seconds for non-info messages
            if message_type != 'info':
                self.root.after(5000, lambda: self.status_label.configure(text="Ready", fg='#ffffff'))
                
        except Exception as e:
            print(f"❌ Status update error: {e}")
    
    def update_time(self):
        """Updates time"""
        try:
            import time
            current_time = time.strftime("%H:%M:%S")
            self.time_label.configure(text=current_time)
            self.root.after(1000, self.update_time)
        except Exception:
            pass
            
    def update_score(self, score: int):
        """Updates score"""
        try:
            self.score = score
            if hasattr(self, 'score_label'):
                self.animate_score_update(score)
        except Exception as e:
            print(f"❌ Score update error: {e}")
    
    def animate_score_update(self, new_score: int):
        """Score update animation"""
        def animate_step(current, target, step=0):
            if step < 10 and current != target:
                increment = (target - current) / (10 - step)
                current += increment
                self.score_label.configure(text=f"Score: {int(current)}")
                
                # Color effect
                colors = ['#e74c3c', '#f39c12', '#2ecc71'] 
                color_index = step % len(colors)
                self.score_label.configure(fg=colors[color_index])
                
                self.root.after(50, lambda: animate_step(current, target, step + 1))
            else:
                self.score_label.configure(text=f"Score: {target}", fg='#e74c3c')
                
        animate_step(self.score, new_score)
        
    def show_error_message(self, title: str, message: str):
        """Shows error message"""
        messagebox.showerror(title, message)
        self.update_status(f"ERROR: {message[:50]}...", 'error')
        
    def show_success_message(self, title: str, message: str):
        """Shows success message"""
        messagebox.showinfo(title, message)
        
    def show_info_message(self, title: str, message: str):
        """Shows info message"""
        messagebox.showinfo(title, message)
        
    def set_camera_callback(self, callback: Callable):
        """Sets camera callback function"""
        self.camera_callback = callback
        
    def set_gesture_callback(self, callback: Callable):
        """Sets gesture callback function"""
        self.gesture_callback = callback
        
    def on_volume_change(self, value):
        """Called when volume level changes"""
        volume = float(value) / 100.0
        self.update_status(f"Volume level: %{int(float(value))}")
        
        # Call callback if exists to notify volume manager
        if hasattr(self, 'volume_callback') and self.volume_callback:
            self.volume_callback(volume)
    
    def on_sensitivity_change(self, value):
        """Called when gesture sensitivity changes"""
        sensitivity = float(value)
        self.gesture_confidence_threshold = sensitivity
        self.update_status(f"Gesture sensitivity: {sensitivity:.1f}")
        
    def on_stability_change(self):
        """Called when stability frames change"""
        stability = self.stability_var.get()
        self.gesture_stability_frames = stability
        self.update_status(f"Stability frames: {stability}")
        # Reset frame counter
        self.gesture_frame_counter = {}
        
    def set_volume_callback(self, callback):
        """Sets volume callback function"""
        self.volume_callback = callback
        
    def set_instrument_callback(self, callback):
        """Sets instrument callback function"""
        self.instrument_callback = callback
        
    def set_beat_challenge_callback(self, callback):
        """Sets beat challenge callback function"""
        self.beat_challenge_callback = callback
        
    def _highlight_pattern_step(self, position: int, sound_id: str, is_active_target: bool = False):
        """Highlights the pattern step and active target."""
        if not (0 <= position < len(self.pattern_display_labels)):
            print(f"⚠️ Invalid position: {position}")
            return

        # Clear all highlights except the current position
        # for i, label in enumerate(self.pattern_display_labels):
        #     if i != position:
        #         label.configure(background=self.theme['secondary_bg_color'], 
        #                         foreground=self.theme.get('secondary_text_color', self.theme['text_color']))

        target_label = self.pattern_display_labels[position]
        
        # Emoji or text
        display_text = self.emoji_map.get(sound_id, self.emoji_map["default"])
        target_label.configure(text=display_text)

        if is_active_target:
            target_label.configure(background=self.theme['accent_color'], 
                                   foreground=self.theme.get('accent_fg_color', 'white'))
            # self._start_pulse_effect(target_label) # Pulse effect for active target
        else:
            # Inactive but "hit-able" step if sound_id is not '0'
            if sound_id != '0':
                target_label.configure(background=self.theme.get('pattern_step_bg', self.theme['secondary_bg_color']), 
                                       foreground=self.theme.get('pattern_step_fg', self.theme['text_color']))
            else: # Empty ('0') step
                target_label.configure(background=self.theme.get('pattern_empty_bg', self.theme['bg_color']), 
                                       foreground=self.theme.get('pattern_empty_fg', self.theme['text_color']))

    def _start_pulse_effect(self, widget):
        """Pulse effect for gesture indicator (bottom panel)"""
        def pulse(step=0):
            if step < 6:  # 3 pulse cycle
                colors = ['#e74c3c', '#c0392b'] if step % 2 == 0 else ['#c0392b', '#e74c3c']
                try:
                    widget.configure(bg=colors[0])
                    self.root.after(150, lambda: pulse(step + 1))
                except:
                    pass  # Widget destroyed
                    
        pulse()
        
    def on_instrument_change_event(self, event):
        """Bind instrument change event"""
        selected_instrument = self.instrument_combo.get()
        self.update_status(f"🎹 Instrument changed: {selected_instrument}", 'info')
        
        # Call callback if exists to notify instrument manager
        if hasattr(self, 'instrument_callback') and self.instrument_callback:
            self.instrument_callback(selected_instrument)
            
        # Current instrument display update
        self.update_current_instrument_display(selected_instrument)
    
    def update_current_instrument_display(self, instrument: str):
        """Updates current instrument display"""
        try:
            # Instrument emojis
            instrument_emojis = {
                'Piano': '🎹',
                'Drums': '🥁', 
                'Synth': '🎛️',
                'Guitar': '🎸'
            }
            
            emoji = instrument_emojis.get(instrument, '🎵')
            self.current_instrument_label.configure(
                text=f"{emoji} {instrument}",
                fg='#e74c3c'  # Accent color
            )
            
            # Normalize color after short time
            self.root.after(1000, lambda: self.current_instrument_label.configure(fg='#3498db'))
            
        except Exception as e:
            print(f"❌ Instrument display update error: {e}")
        
    def run(self):
        """Starts main loop"""
        self.root.mainloop()
        
    def destroy(self):
        """Closes the window"""
        self.is_camera_running = False
        self.root.destroy()

    def _setup_beat_challenge_ui(self):
        """Sets up UI elements for Beat Challenge"""
        challenge_ui_frame = ttk.LabelFrame(self.right_panel, text="🏆 Beat Challenge", padding=10)
        challenge_ui_frame.pack(pady=10, padx=10, fill='x', side='bottom')

        # Skor Label
        self.beat_challenge_score_label = ttk.Label(
            challenge_ui_frame,
            text="Score: 0",
            font=(self.theme['font_family'], 16, 'bold'),
            style='Accent.TLabel'
        )
        self.beat_challenge_score_label.pack(pady=(5, 10))

        # Guide Label (large guide)
        self.beat_challenge_guide_label = ttk.Label(
            challenge_ui_frame,
            text="",
            font=(self.theme['font_family'], 28, 'bold'),
            background=self.theme['bg_color'],
            foreground='#e67e22',
            anchor='center',
            padding=10
        )
        self.beat_challenge_guide_label.pack(pady=(0, 10), fill='x')

        # Combo Label
        self.beat_challenge_combo_label = ttk.Label(
            challenge_ui_frame,
            text="Combo: 0",
            font=(self.theme['font_family'], 14, 'bold'),
            foreground='#f39c12',
            background=self.theme['bg_color']
        )
        self.beat_challenge_combo_label.pack(pady=(0, 10))

        # Feedback Label
        self.beat_challenge_feedback_label = ttk.Label(
            challenge_ui_frame,
            text="",
            font=(self.theme['font_family'], 14, 'italic'),
            anchor="center"
        )
        self.beat_challenge_feedback_label.pack(pady=(0, 10))
        
        # Pattern Display Area
        pattern_display_frame = ttk.Frame(challenge_ui_frame, style='TFrame')
        pattern_display_frame.pack(pady=(5,5), fill='x')
        
        # Maximum 8 step pattern assumed
        self.pattern_display_labels = []
        for i in range(8): # Max 8 pattern steps for placeholder
            label = ttk.Label(pattern_display_frame, text="-", font=(self.theme['font_family'], 12), relief="solid", padding=5, width=5, anchor="center", style='Secondary.TLabel')
            label.pack(side='left', padx=2, expand=True, fill='x')
            self.pattern_display_labels.append(label)

    def _start_beat_challenge_clicked(self):
        """Called when Beat Challenge button is clicked"""
        if self.beat_challenge_callback:
            # Ensure camera is running
            if not self.is_camera_running:
                self.show_error_message("Camera Off", "You must start the camera before starting the Beat Challenge.")
                return
            
            # If challenge is already running, stop, otherwise start
            if self.beat_mode:
                self.beat_challenge_callback(action="stop") # Send stop signal to App
                self.beat_challenge_button.configure(text="🥁 Start Beat Challenge")
                self.update_status("Beat Challenge stopped.", "info")
                self.beat_mode = False # Update state in MainWindow
            else:
                self.beat_challenge_callback(action="start") # Send start signal to App
                self.beat_challenge_button.configure(text="🛑 Stop Beat Challenge")
                self.update_status("Beat Challenge started!", "success")
                self.beat_mode = True # Update state in MainWindow
                self.clear_beat_challenge_feedback() # Clear feedback at start
                self.update_score(0) # Reset score

    def update_beat_challenge_feedback(self, feedback_key: str, score: int, target_position: Optional[int], combo: int = 0):
        """Updates Beat Challenge feedback (Perfect, Good, Miss) and score. Combo is also displayed."""
        if not self.beat_challenge_feedback_label:
            return

        feedback_text_map = {
            "perfect": "🚀 Perfect!",
            "good": "👍 Good!",
            "miss": "😥 Miss...",
            "wrong_sound": "🤔 Wrong Gesture!",
            "early_late": "⏱️ Timing Error!"
        }
        text = feedback_text_map.get(feedback_key, "")
        color_bg, color_fg = self.feedback_colors.get(feedback_key, self.feedback_colors["default"])
        self.beat_challenge_feedback_label.config(text=text, background=color_bg, foreground=color_fg)
        self.update_score(score)
        # Combo update
        if self.beat_challenge_combo_label:
            self.beat_challenge_combo_label.config(text=f"Combo: {combo}")
            # Combo animation (short color effect)
            if combo > 1 and feedback_key in ["perfect", "good"]:
                original_fg = self.beat_challenge_combo_label.cget('foreground')
                self.beat_challenge_combo_label.config(foreground='#2ecc71')
                self.root.after(400, lambda: self.beat_challenge_combo_label.config(foreground=original_fg))
        # Clear feedback after short time
        self.root.after(700, self.clear_beat_challenge_feedback)
        # If there is a target position and not "miss", briefly highlight the step
        if target_position is not None and feedback_key not in ['miss', 'early_late', 'wrong_sound'] and target_position < len(self.pattern_display_labels):
            original_bg = self.pattern_display_labels[target_position].cget('background')
            original_fg = self.pattern_display_labels[target_position].cget('foreground')
            self.pattern_display_labels[target_position].config(background=color_bg, foreground=color_fg)
            def reset_target_style():
                if target_position is not None and 0 <= target_position < len(self.pattern_display_labels):
                    label_to_reset = self.pattern_display_labels[target_position]
                    current_text = label_to_reset.cget("text")
                    found_sound_id = "0"
                    for key, value in self.emoji_map.items():
                        if value == current_text:
                            found_sound_id = key
                            break
                    self._highlight_pattern_step(target_position, found_sound_id, is_active_target=False)
            if target_position is not None and 0 <= target_position < len(self.pattern_display_labels):
                self.pattern_display_labels[target_position].after(300, reset_target_style)

    def clear_beat_challenge_feedback(self):
        """Clears Beat Challenge feedback label."""
        if self.beat_challenge_feedback_label:
            default_bg, default_fg = self.feedback_colors["default"]
            self.beat_challenge_feedback_label.config(text="", background=default_bg, foreground=default_fg)

    def update_beat_challenge_guide(self, guide_text: str):
        """Displays expected gesture and tur information in guide label."""
        if self.beat_challenge_guide_label:
            self.beat_challenge_guide_label.config(
                text=guide_text, # Now use directly received text
                foreground='#e67e22', # Orange color, attention-grabbing
                background=self.theme['bg_color']
            )
    def animate_beat_challenge_guide(self, success: bool):
        """Applies color animation in guide label based on correct/incorrect."""
        if not self.beat_challenge_guide_label:
            return
        color = '#2ecc71' if success else '#e74c3c'
        self.beat_challenge_guide_label.config(background=color, foreground='white')
        self.root.after(600, lambda: self.beat_challenge_guide_label.config(background=self.theme['bg_color'], foreground='#e67e22'))

    def update_simon_says_pattern_display(self, current_pattern: List[str], highlight_index: int = -1, player_inputs: Optional[List[Tuple[str, bool]]] = None, phase: str = 'SHOWING'):
        """Updates pattern display on right panel for Simon Says game."""
        num_display_labels = len(self.pattern_display_labels)

        for i in range(num_display_labels):
            label = self.pattern_display_labels[i]
            if i < len(current_pattern):
                gesture_in_pattern = current_pattern[i]
                emoji_to_show = self.emoji_map.get(gesture_in_pattern, "?")
                # Always update text, because pattern can change or player input can come
                label.config(text=emoji_to_show, relief="solid") 

                if phase == 'SHOWING':
                    if i == highlight_index:
                        # Active step shown by app
                        label.config(background=self.theme['accent_color'], foreground=self.theme.get('accent_fg_color', 'white'), font=(self.theme['font_family'], 14, 'bold'))
                    else:
                        # Not yet shown or past steps (light emoji)
                        label.config(background=self.theme['secondary_bg_color'], foreground=self.theme.get('secondary_text_color', '#bdc3c7'), font=(self.theme['font_family'], 12))
                
                elif phase == 'PLAYER_TURN':
                    if player_inputs and i < len(player_inputs):
                        # Player's input step
                        _gesture, success = player_inputs[i]
                        # Show input gesture's emoji
                        label.config(text=self.emoji_map.get(_gesture, "ERR")) 
                        if success:
                            label.config(background=self.feedback_colors["perfect"][0], foreground=self.feedback_colors["perfect"][1], font=(self.theme['font_family'], 12, 'bold'))
                        else:
                            label.config(background=self.feedback_colors["miss"][0], foreground=self.feedback_colors["miss"][1], font=(self.theme['font_family'], 12, 'bold'))
                    elif i == highlight_index:
                        # Player's next step to BE EXPECTED
                        label.config(text="❓", background=self.theme['accent_color'], foreground=self.theme.get('accent_fg_color', 'white'), font=(self.theme['font_family'], 14, 'bold'), relief="sunken")
                    else:
                        # Player hasn't entered yet, future steps (original emoji, light)
                        label.config(background=self.theme['secondary_bg_color'], foreground=self.theme.get('secondary_text_color', '#bdc3c7'), font=(self.theme['font_family'], 12))
                
                # IDLE, GAME_OVER, LEVEL_COMPLETE phases, show pattern's emojis light if pattern exists
                elif phase in ['IDLE', 'GAME_OVER', 'LEVEL_COMPLETE'] and current_pattern:
                     label.config(text=emoji_to_show, background=self.theme['secondary_bg_color'], foreground=self.theme.get('secondary_text_color', '#7f8c8d'), font=(self.theme['font_family'], 12))
                else: # Nothing matches or pattern is empty default
                    label.config(text="-", background=self.theme['secondary_bg_color'], foreground=self.theme.get('secondary_text_color', '#7f8c8d'), font=(self.theme['font_family'], 12))
            
            else:
                # Extra pattern's boxes to clear (empty and flat)
                label.config(text="", background=self.theme['secondary_bg_color'], relief="flat")


# Test code
if __name__ == "__main__":
    # Simple test
    app = MainWindow()
    
    print("✅ MainWindow class ready!")
    print("📝 Sueda's to implement features:")
    print("   - update_camera_frame() optimization")
    print("   - update_gesture_display() animations")
    print("   - start_beat_challenge() pattern display")
    print("   - Modern UI theme and animations")
    print("   - Error handling and user feedback")
    
    # Test by showing window
    print("🖼️ Test window opening...")
    app.run() 