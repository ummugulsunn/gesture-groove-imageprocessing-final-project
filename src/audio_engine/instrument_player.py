"""
Instrument Player Module - Ümmu Gülsün's Task
Playing different instruments and beat challenge system
"""

import pygame
import numpy as np
import threading
import time
from typing import Dict, List, Optional
from .sound_manager import SoundManager


class InstrumentPlayer:
    """
    Instrument playing and beat challenge system
    
    Features to be developed by Ümmu Gülsün:
    - Multiple instrument support
    - Beat Challenge game mode
    - Rhythm pattern system
    - MIDI note mapping
    - Performance metrics
    """
    
    def __init__(self, sound_manager: SoundManager):
        """
        Initializes InstrumentPlayer
        
        Args:
            sound_manager: SoundManager reference
        """
        self.sound_manager = sound_manager
        
        # Instrument settings
        self.current_instrument = 'piano'
        self.available_instruments = ['piano', 'drums', 'synth', 'guitar']
        
        # Beat Challenge
        self.beat_mode = False
        self.beat_pattern = []
        self.beat_index = 0
        self.bpm = 120
        self.score = 0
        self.hit_accuracy = []
        
        # Performance tracking
        self.total_notes_played = 0
        self.session_start_time = time.time()
        
        # For threading
        self.beat_thread = None
        self.is_beat_playing = False
        
        print("🎹 InstrumentPlayer ready!")
        
    def set_instrument(self, instrument: str):
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Changes active instrument
        
        Args:
            instrument: Instrument name ('piano', 'drums', 'synth', 'guitar')
        """
        if instrument in self.available_instruments:
            self.current_instrument = instrument
            print(f"🎵 Active instrument: {instrument}")
            return True
        else:
            print(f"❌ Invalid instrument: {instrument}")
            return False
    
    def play_gesture_sound(self, gesture: str) -> bool:
        """
        Plays appropriate instrument sound for gesture
        
        Args:
            gesture: Recognized gesture
            
        Returns:
            Successful playback status
        """
        if not gesture:
            return False
            
        # Sound mapping based on gesture
        sound_mapping = {
            'open_hand': self._get_instrument_sound('C4'),
            'fist': self._get_instrument_sound('D4'),
            'point_up': self._get_instrument_sound('E4'),
            'point_down': self._get_instrument_sound('F4'),
            'peace': self._get_instrument_sound('G4')
        }
        
        if gesture in sound_mapping:
            success = self.sound_manager.play_sound(gesture)
            
            if success:
                self.total_notes_played += 1
                print(f"🎵 {self.current_instrument}: {gesture}")
                
                # Accuracy check in beat mode
                if self.beat_mode:
                    self._check_beat_accuracy(gesture)
                    
                return True
                
        return False
    
    def _get_instrument_sound(self, note: str) -> str:
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Returns sound ID for instrument and note combination
        
        Args:
            note: MIDI note name (e.g: 'C4', 'A#3')
            
        Returns:
            Sound file ID
        """
        # TODO: To be implemented by Ümmu Gülsün
        # Instrument-note mapping system
        return f"{self.current_instrument}_{note}"
    
    def start_beat_challenge(self, difficulty: str = 'easy'):
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Starts Beat Challenge mode
        
        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')
        """
        self.beat_mode = True
        self.score = 0
        self.beat_index = 0
        self.hit_accuracy = []
        
        # Select pattern based on difficulty
        patterns = {
            'easy': ['open_hand', 'fist', 'open_hand', 'fist'],
            'medium': ['open_hand', 'fist', 'point_up', 'open_hand', 'fist', 'point_up'],
            'hard': ['open_hand', 'fist', 'point_up', 'point_down', 'peace', 'open_hand']
        }
        
        self.beat_pattern = patterns.get(difficulty, patterns['easy'])
        
        # Set BPM
        bpm_settings = {'easy': 100, 'medium': 120, 'hard': 140}
        self.bpm = bpm_settings.get(difficulty, 120)
        
        print(f"🎯 Beat Challenge started! Difficulty: {difficulty}")
        print(f"🎵 Pattern: {' -> '.join(self.beat_pattern)}")
        print(f"⏱️ BPM: {self.bpm}")
        
        # Start beat thread
        self._start_beat_thread()
        
    def _start_beat_thread(self):
        """Timing thread for Beat Challenge"""
        if self.beat_thread and self.beat_thread.is_alive():
            return
            
        self.is_beat_playing = True
        self.beat_thread = threading.Thread(target=self._beat_loop, daemon=True)
        self.beat_thread.start()
        
    def _beat_loop(self):
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Beat timing loop
        """
        beat_interval = 60.0 / self.bpm  # Beat interval in seconds
        
        while self.is_beat_playing and self.beat_mode:
            # Show expected gesture (send signal to UI)
            expected_gesture = self.beat_pattern[self.beat_index % len(self.beat_pattern)]
            
            # Play metronome sound (optional)
            # self.sound_manager.play_sound('metronome_tick')
            
            print(f"🎯 Expected gesture: {expected_gesture}")
            
            # Wait for beat interval
            time.sleep(beat_interval)
            
            # Move to next beat
            self.beat_index += 1
            
            # Return to start if pattern completed
            if self.beat_index >= len(self.beat_pattern) * 3:  # Repeat 3 times
                self.stop_beat_challenge()
                break
                
    def _check_beat_accuracy(self, gesture: str):
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Beat accuracy check
        
        Args:
            gesture: User's gesture
        """
        if not self.beat_mode:
            return
            
        expected = self.beat_pattern[self.beat_index % len(self.beat_pattern)]
        
        # Calculate timing accuracy
        current_time = time.time()
        beat_time = self.session_start_time + (self.beat_index * (60.0 / self.bpm))
        timing_diff = abs(current_time - beat_time)
        
        # Score based on accuracy
        if gesture == expected:
            if timing_diff < 0.1:  # Perfect timing
                self.score += 100
                accuracy = 1.0
            elif timing_diff < 0.2:  # Good timing
                self.score += 50
                accuracy = 0.8
            else:  # Late but correct
                self.score += 20
                accuracy = 0.5
        else:
            self.score -= 10  # Wrong gesture penalty
            accuracy = 0.0
            
        self.hit_accuracy.append(accuracy)
        print(f"🎯 Accuracy: {accuracy:.2f}, Score: {self.score}")
        
    def stop_beat_challenge(self):
        """Stops Beat Challenge mode"""
        self.beat_mode = False
        self.is_beat_playing = False
        
        if self.beat_thread and self.beat_thread.is_alive():
            self.beat_thread.join(timeout=1.0)
            
        print(f"🏁 Beat Challenge ended! Final score: {self.score}")
        print(f"📊 Average accuracy: {sum(self.hit_accuracy) / len(self.hit_accuracy):.2f}")
        
    def create_custom_pattern(self, gestures: List[str]):
        """
        Creates custom beat pattern
        
        Args:
            gestures: List of gestures for pattern
        """
        if gestures:
            self.beat_pattern = gestures
            print(f"🎵 Custom pattern created: {' -> '.join(gestures)}")
            return True
        return False
        
    def play_chord(self, notes: List[str]):
        """
        Plays chord with multiple notes
        
        Args:
            notes: List of note names
        """
        for note in notes:
            self.sound_manager.play_note(self.current_instrument, note)
            
    def add_rhythm_effects(self, effect_type: str):
        """
        Adds rhythm effects to current pattern
        
        Args:
            effect_type: Effect type ('reverb', 'delay', 'echo')
        """
        # TODO: Implement rhythm effects
        print(f"🎛️ Adding {effect_type} effect to pattern")
        
    def record_session(self):
        """Records current session for playback"""
        # TODO: Implement session recording
        print("🎥 Recording session...")
        
    def get_performance_stats(self) -> Dict:
        """
        Returns performance statistics
        
        Returns:
            Dictionary containing performance metrics
        """
        session_duration = time.time() - self.session_start_time
        
        return {
            'total_notes': self.total_notes_played,
            'session_duration': session_duration,
            'notes_per_minute': (self.total_notes_played / session_duration) * 60 if session_duration > 0 else 0,
            'average_accuracy': sum(self.hit_accuracy) / len(self.hit_accuracy) if self.hit_accuracy else 0,
            'current_score': self.score
        }
        
    def reset_stats(self):
        """Resets performance statistics"""
        self.total_notes_played = 0
        self.session_start_time = time.time()
        self.score = 0
        self.hit_accuracy = []
        print("📊 Statistics reset")


# Test code
if __name__ == "__main__":
    # Simple test
    sound_mgr = SoundManager()
    player = InstrumentPlayer(sound_mgr)
    
    print("✅ InstrumentPlayer class ready!")
    print("📝 Ümmu Gülsün's features to be implemented:")
    print("   - set_instrument()")
    print("   - _get_instrument_sound()")
    print("   - start_beat_challenge()")
    print("   - _beat_loop()")
    print("   - _check_beat_accuracy()")
    
    # Test basic functionality
    player.set_instrument('piano')
    player.play_gesture_sound('open_hand')
    
    time.sleep(1)
    print("✅ Test completed!") 