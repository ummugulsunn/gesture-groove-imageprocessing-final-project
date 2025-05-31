#!/usr/bin/env python3
"""
GestureGroove Main Application
Application that plays music with hand gestures

Development Team: Ümmü Gülsün
Course: Image Processing
"""

import sys
import os
import cv2
import time
import threading
from typing import Optional, List, Tuple
import random

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gesture_recognition import HandDetector, GestureClassifier
from audio_engine import SoundManager, InstrumentPlayer
from ui import MainWindow
from utils.config import get_config, create_directories


class GestureGrooveApp:
    """
    Main application class - combines all modules
    """
    
    def __init__(self):
        """Initializes the application"""
        print("🎵 Starting GestureGroove...")
        
        # Load configuration
        self.config = get_config()
        create_directories()
        
        # Beat Challenge - Progressive Pattern Repetition (Simon Says) state variables
        self.beat_challenge_game_phase = 'IDLE'  # Possible states: IDLE, SHOWING_PATTERN, PLAYER_TURN, LEVEL_COMPLETE, GAME_OVER
        self.beat_challenge_current_level = 1
        self.beat_challenge_current_pattern: List[str] = []
        self.beat_challenge_pattern_show_index = 0  # Used when application shows pattern
        self.beat_challenge_player_input_index = 0  # Used when user enters pattern
        self.beat_challenge_score = 0
        
        self.beat_challenge_pattern_base_length = 2  # Pattern length in first level
        self.beat_challenge_show_step_delay = 1.2  # Seconds: Time to show each pattern step
        self.beat_challenge_level_complete_delay = 1.5  # Seconds: Wait time after level completion
        self.player_inputs_for_ui: List[Tuple[str, bool]] = []  # Store player inputs for UI

        # Simon Says - Player Input Control Timer
        self.player_input_collection_timer: Optional[threading.Timer] = None
        self.player_action_time_limit: float = 2.0  # Seconds: Time allowed for player to make gesture
        self.player_turn_input_received: bool = False  # Whether valid input received for current step

        self.available_gestures = ['fist', 'point_up', 'open_hand', 'peace', 'point_down']  # Available gestures
        self.original_instrument_before_challenge = ""  # Store instrument before challenge

        # Old Beat Challenge parameters (for Reaction game) - To be cleaned or adapted if needed
        # self.beat_challenge_reaction_time = 1.5 
        # self.beat_challenge_pause_between_steps = 0.5
        # self.beat_challenge_max_rounds = 10
        
        # Initialize core components
        self.init_components()
        
        # Initialize UI
        self.init_ui()
        
        # Application state
        self.is_running = False
        self.camera = None
        self.current_gesture = None  # Only for normal mode instant gesture, challenge uses its own state
            
        print("✅ GestureGroove ready!")
            
    def init_components(self):
        """Initializes core components"""
        try:
            # Hand detection system (Ümmü Gülsün's module)
            self.hand_detector = HandDetector(
                max_num_hands=self.config['hand_detection']['max_num_hands'],
                min_detection_confidence=self.config['hand_detection']['min_detection_confidence'],
                min_tracking_confidence=self.config['hand_detection']['min_tracking_confidence']
            )
            print("✅ Hand detection system ready")
            
            # Gesture classifier (Ümmü Gülsün's module)
            self.gesture_classifier = GestureClassifier()
            print("✅ Gesture classifier ready")
            
            # Sound management (Ümmu Gülsün's module)
            self.sound_manager = SoundManager(
                frequency=self.config['audio']['frequency'],
                buffer_size=self.config['audio']['buffer_size'],
                channels=self.config['audio']['channels']
            )
            print("✅ Sound system ready")
            
            # Instrument player (Ümmu Gülsün's module)
            self.instrument_player = InstrumentPlayer(self.sound_manager)
            print("✅ Instrument system ready")
            
            # Load real instrument sounds
            self.generate_test_sounds()
            
        except Exception as e:
            print(f"❌ Component initialization error: {e}")
            sys.exit(1)
            
    def init_ui(self):
        """Initializes UI (Sueda's module)"""
        try:
            self.main_window = MainWindow()
            
            # Set up callbacks
            self.main_window.set_camera_callback(self.camera_loop)
            self.main_window.set_volume_callback(self.on_volume_change)
            self.main_window.set_instrument_callback(self.on_instrument_change)
            self.main_window.set_beat_challenge_callback(self.toggle_beat_challenge)
            
            print("✅ User interface ready")
            
        except Exception as e:
            print(f"❌ UI initialization error: {e}")
            sys.exit(1)
            
    def generate_test_sounds(self):
        """Load real instrument sounds"""
        print("🎵 Loading real instrument sounds...")
        
        # Load instrument packs
        instruments = ['piano', 'drums', 'synth', 'guitar']
        for instrument in instruments:
            success = self.sound_manager.load_instrument_pack(instrument)
            if success:
                print(f"✅ {instrument} sounds loaded")
            else:
                print(f"❌ Failed to load {instrument} sounds")
        
        # Set up gesture-to-sound mapping
        self.gesture_sound_mapping = {
            'piano': {
                'open_hand': 'C4',
                'fist': 'D4', 
                'point_up': 'E4',
                'point_down': 'F4',
                'peace': 'G4'
            },
            'drums': {
                'open_hand': 'kick',
                'fist': 'snare',
                'point_up': 'hihat', 
                'point_down': 'crash',
                'peace': 'ride'
            },
            'synth': {
                'open_hand': 'synth_C4',
                'fist': 'synth_D4',
                'point_up': 'synth_E4',
                'point_down': 'synth_F4', 
                'peace': 'synth_G4'
            },
            'guitar': {
                'open_hand': 'C4',  # Now real guitar sounds
                'fist': 'D4',
                'point_up': 'E4',
                'point_down': 'F4',
                'peace': 'G4'
            }
        }
        
        # Default instrument
        self.current_instrument = 'piano'
            
    def camera_loop(self):
        """
        Main camera loop
        Real-time hand detection and gesture recognition
        """
        # Initialize camera
        self.camera = cv2.VideoCapture(self.config['camera']['device_id'])
        
        if not self.camera.isOpened():
            print("❌ Could not open camera!")
            return
            
        # Camera settings
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        print("📷 Camera initialized")
        
        # FPS calculation
        prev_time = time.time()
        
        while self.main_window.is_camera_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Could not read camera frame!")
                break
                
            # Flip the image (selfie mode)
            frame = cv2.flip(frame, 1)
            
            # Detect hands (Ümmü Gülsün's module)
            processed_frame, hand_landmarks = self.hand_detector.detect_hands(frame)
            
            # Gesture recognition
            gesture = None
            opencv_gesture = None
            confidence = 0.0
            extracted_landmarks = None
            
            if hand_landmarks:
                # Get the first hand
                landmarks = self.hand_detector.extract_landmarks(hand_landmarks[0])
                extracted_landmarks = landmarks  # Store for UI
                
                # If OpenCV detection, directly get gesture
                if not self.hand_detector.use_mediapipe:
                    opencv_gesture = self.hand_detector._advanced_gesture_inference(hand_landmarks[0], processed_frame)
                
                # Gesture classification (if model trained or OpenCV gesture detected)
                gesture = self.gesture_classifier.predict_gesture(landmarks, opencv_gesture)
                
                # Calculate confidence (simple approximation)
                if gesture:
                    confidence = 0.8 if self.hand_detector.use_mediapipe else 0.6
                    
            # Play sound if gesture changed
            if gesture != self.current_gesture and gesture and gesture != 'none':
                self.play_gesture_sound(gesture)
                self.current_gesture = gesture
                
            # Update UI (Sueda's module)
            self.main_window.update_camera_frame(processed_frame)
            self.main_window.update_gesture_display(gesture, confidence)
            
            # Update Hand landmarks visualization
            if extracted_landmarks is not None:
                self.main_window.update_hand_landmarks(extracted_landmarks)
            else:
                self.main_window.update_hand_landmarks(None)
            
            # Calculate FPS and update
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != curr_time else 0
            prev_time = curr_time
            self.main_window.update_fps(fps)
            
            # Short wait (to reduce CPU usage)
            time.sleep(0.01)
            
        # Close camera
        if self.camera:
            self.camera.release()
            print("📷 Camera closed")
            
    def play_gesture_sound(self, gesture: str):
        """Plays sound for gesture (Progressive Pattern Repetition - Simon Says logic)"""
        
        # First, play sound in every state (to allow user to "follow" with selected instrument)
        # This allows to play sound in SHOWING_PATTERN state
        sound_map_for_current_instrument = self.gesture_sound_mapping.get(self.current_instrument, {})
        played_sound_id = sound_map_for_current_instrument.get(gesture)
        
        # Play sound in every state, but only if gesture in mapping
        if played_sound_id:
            self.sound_manager.play_instrument_sound(self.current_instrument, played_sound_id)

        # BEAT CHALLENGE MODU CONTROL
        if self.main_window.beat_mode:
            if self.beat_challenge_game_phase == 'PLAYER_TURN':
                if not self.player_turn_input_received: # If no input processed for current step
                    if gesture and gesture != 'none': # If valid gesture detected
                        if self.player_input_collection_timer: # And timer still active (i.e., time not expired)
                            print(f"▶️ Player timed out '{gesture}' gesture (Step: {self.beat_challenge_player_input_index + 1}). Timer canceled.")
                            self.player_input_collection_timer.cancel()
                            self.player_input_collection_timer = None
                            self.player_turn_input_received = True
                            self._evaluate_player_gesture(gesture) 
                        # else: Timer not active (time expired and timeout function active) or already input received, do nothing.
                    # else: gesture 'none' or None, player didn't make any gesture, timeout handles.
                # else: Input already received for this step (either on time or timeout)
            
            # Other phases (SHOWING_PATTERN, LEVEL_COMPLETE, GAME_OVER, IDLE) player's instant gesture score doesn't affect.
            # Sound already played above (in every state)
            return # Beat challenge mode, normal sound logging

        # NORMAL SOUND PLAY (Beat Challenge not active)
        # This part remains the same, but will run after the ABOVE BEAT_CHALLENGE block
        # Zaten `played_sound_id` and `self.current_instrument` above are defined and sound played
        # Only for console logging:
        if played_sound_id and not self.main_window.beat_mode:
             print(f"🎵 {self.current_instrument}: {gesture} -> {played_sound_id}")
        elif not self.main_window.beat_mode: # Only log if normal mode and no mapping
            # print(f"❌ Gesture mapping not found: {self.current_instrument} / {gesture}") # Too much log
            pass 

    def _evaluate_player_gesture(self, player_gesture: str):
        """Evaluates player's (timed or expired) gesture."""
        if not self.main_window.beat_mode or self.beat_challenge_game_phase != 'PLAYER_TURN':
            print("⚠️ Evaluation not in correct game phase or game stopped.")
            return

        if self.beat_challenge_player_input_index >= len(self.beat_challenge_current_pattern):
            print("⚠️ Evaluation index error!")
            return

        expected_gesture = self.beat_challenge_current_pattern[self.beat_challenge_player_input_index]

        if player_gesture == "timeout_miss":
            # Time expired and no valid gesture detected.
            # UI's gesture to show, could be last stable gesture on camera or "?"
            actual_gesture_on_timeout = self.main_window.current_gesture
            display_gesture_for_ui = actual_gesture_on_timeout if actual_gesture_on_timeout and actual_gesture_on_timeout != 'none' else "❓"
            
            self.player_inputs_for_ui.append((display_gesture_for_ui, False)) 
            self.main_window.animate_beat_challenge_guide(success=False)
            self.beat_challenge_game_phase = 'GAME_OVER'
            self.main_window.update_simon_says_pattern_display(
                current_pattern=self.beat_challenge_current_pattern,
                highlight_index=self.beat_challenge_player_input_index, 
                player_inputs=self.player_inputs_for_ui,
                phase='PLAYER_TURN' 
            )
            self.main_window.update_beat_challenge_guide(f"Time Expired! ⌛") # Short message
            # stop_beat_challenge'ı short wait call to let user see message
            threading.Timer(1.0, lambda: self.stop_beat_challenge(final=True, message_prefix=f"Time Expired! Level {self.beat_challenge_current_level}'d Failed.")).start()
            return

        if player_gesture == expected_gesture:
            self.beat_challenge_score += 10 * self.beat_challenge_current_level
            self.main_window.update_score(self.beat_challenge_score)
            self.player_inputs_for_ui.append((player_gesture, True))
            self.main_window.update_simon_says_pattern_display(
                current_pattern=self.beat_challenge_current_pattern,
                highlight_index=self.beat_challenge_player_input_index + 1, 
                player_inputs=self.player_inputs_for_ui,
                phase='PLAYER_TURN'
            )
            self.main_window.animate_beat_challenge_guide(success=True)
            self.beat_challenge_player_input_index += 1

            if self.beat_challenge_player_input_index == len(self.beat_challenge_current_pattern):
                self.beat_challenge_game_phase = 'LEVEL_COMPLETE'
                self.main_window.update_beat_challenge_guide(f"Great! Level {self.beat_challenge_current_level} Completed!")
                self.beat_challenge_current_level += 1
                # player_turn_input_received not important here, _prepare_next_level will reset.
                threading.Timer(self.beat_challenge_level_complete_delay, self._prepare_next_level).start()
            else:
                self.main_window.update_beat_challenge_guide(f"Correct! ({self.beat_challenge_player_input_index + 1}/{len(self.beat_challenge_current_pattern)}): ?")
                self._start_player_gesture_collection() # Start timer for next gesture
        else:
            # Incorrect gesture (but made within time)
            self.player_inputs_for_ui.append((player_gesture, False))
            self.main_window.animate_beat_challenge_guide(success=False)
            self.beat_challenge_game_phase = 'GAME_OVER'
            self.main_window.update_simon_says_pattern_display(
                current_pattern=self.beat_challenge_current_pattern,
                highlight_index=self.beat_challenge_player_input_index,
                player_inputs=self.player_inputs_for_ui,
                phase='PLAYER_TURN'
            )
            self.main_window.update_beat_challenge_guide(f"Incorrect Gesture! ❌") # Short message
            threading.Timer(1.0, lambda: self.stop_beat_challenge(final=True, message_prefix=f"Incorrect Gesture! Level {self.beat_challenge_current_level}'d Failed.")).start()
            return # Ensure return from function

    def run(self):
        """Runs the application"""
        try:
            print("🚀 Starting GestureGroove...")
            print("📋 Usage:")
            print("   - Press 'Start Camera' button to start camera")
            print("   - Make hand gestures in front of camera")
            print("   - Different gestures produce different sounds")
            print("   - Try Beat Challenge!")
            print("   - Exit by closing window")
            
            # Start main UI loop
            self.main_window.run()
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"❌ Application error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleans up resources"""
        try:
            if self.camera and self.camera.isOpened():
                self.camera.release()
                
            cv2.destroyAllWindows()
            
            if hasattr(self, 'sound_manager'):
                self.sound_manager.stop_all_sounds()
                
            print("✅ Resources cleaned")
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

    def on_volume_change(self, volume: float):
        """Called when volume changed"""
        self.sound_manager.set_master_volume(volume)
        print(f"🔊 Volume set: {volume:.2f}")
        
    def on_instrument_change(self, instrument: str):
        """Called when instrument changed"""
        instrument_lower = instrument.lower()
        if instrument_lower in self.gesture_sound_mapping:
            self.current_instrument = instrument_lower
            print(f"🎹 Instrument changed: {instrument}")
            
            # UI feedback
            self.main_window.update_status(f"🎹 Instrument: {instrument}", 'success')
            
            # Test sound play - use instrument sounds
            if instrument_lower in self.gesture_sound_mapping:
                test_sound = list(self.gesture_sound_mapping[instrument_lower].values())[0]
                success = self.sound_manager.play_instrument_sound(instrument_lower, test_sound)
                if success:
                    print(f"🎵 Test sound: {instrument_lower}/{test_sound}")
                else:
                    print(f"❌ Test sound play error: {instrument_lower}/{test_sound}")
        else:
            print(f"❌ Unknown instrument: {instrument}")
            self.main_window.update_status(f"❌ Unknown instrument: {instrument}", 'error')
            
    def toggle_beat_challenge(self):
        """New function for Beat Challenge start/stop button."""
        if not self.main_window.is_camera_running:
            self.main_window.show_error_message("Camera Off", "You must start camera before Beat Challenge.")
            return

        if self.main_window.beat_mode: # Game already running, stop
            self.stop_beat_challenge(final=False) # User stopped
        else: # Start game
            self.start_beat_challenge()
            
    def start_beat_challenge(self):
        """Starts Beat Challenge (Progressive Pattern Repetition - Simon Says)"""
        try:
            print("🎯 Starting Beat Challenge (Simon Says)...")
            self.beat_challenge_score = 0
            self.beat_challenge_current_level = 1
            self.beat_challenge_game_phase = 'IDLE' # _prepare_next_level will make SHOWING_PATTERN
            self.main_window.update_score(self.beat_challenge_score)
            self.player_inputs_for_ui = [] # Reset player inputs for each new game
            
            self.main_window.beat_mode = True
            if not self.original_instrument_before_challenge: # Check if restarting after already in a challenge
                self.original_instrument_before_challenge = self.current_instrument
            # User's selected instrument will continue, we don't change it.
            
            self.main_window.update_current_instrument_display(f"{self.current_instrument.title()} (Simon Says)")
            self.main_window.beat_challenge_button.configure(text="🛑 Challenge Stop")
            self.main_window.update_status("Simon Says Started!", "success")
            
            self._prepare_next_level() # Start first level and pattern show process
            
        except Exception as e:
            print(f"❌ Beat Challenge (Simon Says) error: {e}")
            self.main_window.show_error_message("Beat Challenge Error", str(e))
            self.stop_beat_challenge(final=False) # Safe close in case of error

    def _prepare_next_level(self):
        """Creates and starts pattern for next level."""
        if not self.main_window.beat_mode: # If challenge stopped, prepare new level
            return

        print(f"🌟 Level {self.beat_challenge_current_level} preparing...")
        self.beat_challenge_game_phase = 'SHOWING_PATTERN'
        self.beat_challenge_pattern_show_index = 0
        self.beat_challenge_player_input_index = 0
        self.player_inputs_for_ui = [] # Reset player inputs for new level
        
        pattern_length = self.beat_challenge_pattern_base_length + self.beat_challenge_current_level - 1
        self.beat_challenge_current_pattern = [random.choice(self.available_gestures) for _ in range(pattern_length)]
        
        # UI to show pattern (optional, for now just updating rehber label)
        # self.main_window.display_beat_pattern_for_simon(self.beat_challenge_current_pattern) 
        
        self.main_window.update_beat_challenge_guide(f"Level {self.beat_challenge_current_level} - Watch!")
        self.main_window.update_simon_says_pattern_display(self.beat_challenge_current_pattern, highlight_index=-1, phase='SHOWING') # Start with all pattern transparent
        
        # Pattern show process in separate thread
        threading.Thread(target=self._show_pattern_loop, daemon=True).start()

    def _show_pattern_loop(self):
        """Shows pattern to user step by step."""
        import time # time module here import is safer in thread
        time.sleep(0.5) # Short wait for watch message to show

        for gesture_to_show in self.beat_challenge_current_pattern:
            if not self.main_window.beat_mode or self.beat_challenge_game_phase != 'SHOWING_PATTERN':
                print("Pattern show stopped (mode changed).")
                return # Challenge stopped or phase changed, exit loop

            self.main_window.update_beat_challenge_guide(f"Watch ({self.beat_challenge_pattern_show_index + 1}/{len(self.beat_challenge_current_pattern)}): {self.main_window.emoji_map.get(gesture_to_show, gesture_to_show)}")
            self.main_window.update_simon_says_pattern_display(
                current_pattern=self.beat_challenge_current_pattern,
                highlight_index=self.beat_challenge_pattern_show_index,
                phase='SHOWING'
            )
            
            # Play sound (with user's selected instrument)
            sound_map = self.gesture_sound_mapping.get(self.current_instrument, {})
            sound_id_to_play = sound_map.get(gesture_to_show)
            if sound_id_to_play:
                self.sound_manager.play_instrument_sound(self.current_instrument, sound_id_to_play)
            
            time.sleep(self.beat_challenge_show_step_delay)
            self.beat_challenge_pattern_show_index += 1
        
        # Pattern show finished
        if not self.main_window.beat_mode: return # If stopped during this

        # Wait for player to prepare and show message
        self.main_window.update_beat_challenge_guide(f"Ready! (Level {self.beat_challenge_current_level})")
        time.sleep(1.5) # 1.5 seconds preparation time

        if not self.main_window.beat_mode: return # Preparation may stopped

        # Show "Start!" message to user and add short extra time
        self.main_window.update_beat_challenge_guide(f"Start! (Level {self.beat_challenge_current_level})")
        time.sleep(0.7) # Extra 0.7 seconds wait after "Start!" message

        if not self.main_window.beat_mode: return # This extra wait may stopped

        self.beat_challenge_game_phase = 'PLAYER_TURN'
        self.main_window.update_beat_challenge_guide(f"Start! ({self.beat_challenge_player_input_index + 1}/{len(self.beat_challenge_current_pattern)}): ?")
        self.main_window.update_simon_says_pattern_display(
            current_pattern=self.beat_challenge_current_pattern,
            highlight_index=0, # Player's first input step
            player_inputs=[], # No player input yet
            phase='PLAYER_TURN'
        )
        self._start_player_gesture_collection() # Start player gesture collection process

    def _start_player_gesture_collection(self):
        """Starts timer for player gesture input."""
        if not self.main_window.beat_mode or self.beat_challenge_game_phase != 'PLAYER_TURN':
            return

        self.player_turn_input_received = False # Reset for each new step
        
        # Cancel previous timer (for safety, if any)
        if self.player_input_collection_timer:
            self.player_input_collection_timer.cancel()
            self.player_input_collection_timer = None # Clear reference

        self.player_input_collection_timer = threading.Timer(
            self.player_action_time_limit, 
            self._process_player_action_timeout
        )
        self.player_input_collection_timer.start()
        print(f"⏰ Waiting for player {self.player_action_time_limit}s for gesture (Step: {self.beat_challenge_player_input_index + 1})...")

    def _process_player_action_timeout(self):
        """Called when player's gesture making time expired."""
        if not self.main_window.beat_mode or self.beat_challenge_game_phase != 'PLAYER_TURN':
            # Timer triggered but game already ended or switched to different phase
            return

        if self.player_turn_input_received:
            # Player already made gesture on time (via play_gesture_sound)
            return

        print(f"⌛ Time expired! Step: {self.beat_challenge_player_input_index + 1}. Player's current stable gesture (if any) taken.")
        
        # Clear timer reference to make it inactive (not to restart, just for reference)
        # Timer already triggered once, don't start() again unless needed.
        self.player_input_collection_timer = None 

        current_stable_gesture = self.main_window.current_gesture # MainWindow's stable gesture
        # If current_gesture 'none' or None, mark this as special "timeout_miss"
        gesture_to_evaluate = current_stable_gesture if current_stable_gesture and current_stable_gesture != 'none' else "timeout_miss"
        
        self.player_turn_input_received = True # Time expired, this step processed
        self._evaluate_player_gesture(gesture_to_evaluate)

    def stop_beat_challenge(self, final=False, message_prefix=""):
        """Stops Beat Challenge (Simon Says mode)"""
        try:
            was_beat_mode = self.main_window.beat_mode 
            self.main_window.beat_mode = False 
            self.beat_challenge_game_phase = 'GAME_OVER' # Or 'IDLE', final status based on
            
            if was_beat_mode: # Only if game was actually active and now stopped UI update
                 self.main_window.beat_challenge_button.configure(text="🎯 Challenge Start")
                 self.main_window.update_status("Simon Says Stopped.", "info")

            if hasattr(self, 'original_instrument_before_challenge') and self.original_instrument_before_challenge:
                self.current_instrument = self.original_instrument_before_challenge
                self.main_window.update_current_instrument_display(self.current_instrument.title())
                self.original_instrument_before_challenge = "" # Cleanup
            
            if final and was_beat_mode: 
                score = getattr(self, 'beat_challenge_score', 0)
                level_reached = self.beat_challenge_current_level -1 if self.beat_challenge_game_phase != 'LEVEL_COMPLETE' else self.beat_challenge_current_level 
                level_reached = max(1, level_reached) # At least 1. level reached. If didn't even pass first level.
                
                final_message = f"{message_prefix}\nReached Level: {level_reached}\nTotal Score: {score}"
                if not message_prefix: # If no special message (e.g., user stopped themselves)
                    final_message = f"Simon Says Finished!\nReached Level: {level_reached}\nTotal Score: {score}"

                self.main_window.show_info_message("🏆 Simon Says Result 🏆", final_message)
            
            print(f"🛑 Beat Challenge (Simon Says) stopped. Level: {self.beat_challenge_current_level-1}, Score: {self.beat_challenge_score}")
            self.beat_challenge_current_pattern = [] # Reset pattern
            self.main_window.update_beat_challenge_guide("") # Reset rehber label
            self.main_window.update_simon_says_pattern_display([], phase='IDLE') # Reset UI pattern display
            
            # Stop timer and clear
            if self.player_input_collection_timer:
                self.player_input_collection_timer.cancel()
                self.player_input_collection_timer = None
            self.player_turn_input_received = False # Reset status variable
            
            self.beat_challenge_game_phase = 'IDLE' # Set final status as IDLE
        except Exception as e:
            print(f"❌ Beat Challenge (Simon Says) stop error: {e}")
            self.main_window.beat_challenge_button.configure(text="🎯 Challenge Start")
            self.beat_challenge_game_phase = 'IDLE'


def main():
    """Main function"""
    print("=" * 60)
    print("🎵 GestureGroove: El Hareketleriyle Müzik Çalan Uygulama")
    print("📚 Image Processing Dersi Projesi")
    print("👩‍💻 Geliştirici: Ümmü Gülsün")
    print("=" * 60)
    
    try:
        # Uygulamayı oluştur ve çalıştır
        app = GestureGrooveApp()
        app.run()
        
    except Exception as e:
        print(f"❌ Ana uygulama hatası: {e}")
        sys.exit(1)
        
    print("👋 GestureGroove kapatıldı. Görüşmek üzere!")


if __name__ == "__main__":
    main() 