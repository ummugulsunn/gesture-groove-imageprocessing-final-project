"""
Sound Generator - Real sound asset generation
Tool to be used by Ümmu Gülsün for creating sound files
"""

import numpy as np
import scipy.io.wavfile as wavfile
import os
from typing import Dict, List, Tuple
import math


class SoundGenerator:
    """
    Tool for creating high-quality sound files
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def generate_piano_note(self, frequency: float, duration: float = 1.0) -> np.ndarray:
        """Generate piano-like sound (with harmonics)"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Fundamental frequency + harmonics
        fundamental = np.sin(2 * np.pi * frequency * t)
        harmonic2 = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
        harmonic3 = 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
        harmonic4 = 0.125 * np.sin(2 * np.pi * frequency * 4 * t)
        
        # Combine harmonics
        sound = fundamental + harmonic2 + harmonic3 + harmonic4
        
        # ADSR Envelope (Attack, Decay, Sustain, Release)
        attack_time = 0.1
        decay_time = 0.2
        sustain_level = 0.7
        release_time = 0.3
        
        sound = self._apply_adsr(sound, attack_time, decay_time, sustain_level, release_time, duration)
        
        return self._normalize_audio(sound)
    
    def generate_drum_hit(self, drum_type: str, duration: float = 0.8) -> np.ndarray:
        """Generate drum sound"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if drum_type == "kick":
            # Bass drum - low frequency + noise
            base_freq = 60
            sound = np.sin(2 * np.pi * base_freq * t)
            sound += 0.3 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            # Pitch bend down
            pitch_env = np.exp(-8 * t)
            sound = np.sin(2 * np.pi * base_freq * pitch_env * t)
            
        elif drum_type == "snare":
            # Snare - mid freq + noise
            base_freq = 200
            sound = 0.5 * np.sin(2 * np.pi * base_freq * t)
            # Noise component
            noise = 0.7 * np.random.normal(0, 1, len(t))
            sound += noise
            
        elif drum_type == "hihat":
            # Hi-hat - high freq noise
            sound = np.random.normal(0, 1, len(t))
            # High-pass filter effect
            sound = np.diff(np.concatenate(([0], sound)))
            
        elif drum_type == "crash":
            # Crash cymbal - wide spectrum
            freqs = [400, 800, 1200, 2400, 4800]
            sound = np.zeros(len(t))
            for freq in freqs:
                sound += 0.2 * np.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi)
            sound += 0.4 * np.random.normal(0, 1, len(t))
            
        elif drum_type == "ride":
            # Ride cymbal - metallic
            base_freq = 800
            sound = np.sin(2 * np.pi * base_freq * t)
            sound += 0.3 * np.sin(2 * np.pi * base_freq * 1.5 * t)
            sound += 0.2 * np.random.normal(0, 1, len(t))
        
        else:
            sound = np.zeros(len(t))
        
        # Sharp attack envelope
        attack_env = np.exp(-15 * t)
        sound *= attack_env
        
        return self._normalize_audio(sound)
    
    def generate_synth_note(self, frequency: float, wave_type: str = "sawtooth", duration: float = 1.0) -> np.ndarray:
        """Generate synth sound"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if wave_type == "sawtooth":
            # Sawtooth wave
            sound = 2 * (t * frequency - np.floor(t * frequency + 0.5))
            
        elif wave_type == "square":
            # Square wave
            sound = np.sign(np.sin(2 * np.pi * frequency * t))
            
        elif wave_type == "triangle":
            # Triangle wave
            sound = 2 * np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi
            
        elif wave_type == "sine":
            # Clean sine
            sound = np.sin(2 * np.pi * frequency * t)
            
        else:
            sound = np.sin(2 * np.pi * frequency * t)
        
        # Simple implementation for low-pass filter
        sound = self._low_pass_filter(sound, cutoff_freq=2000)
        
        # Synth ADSR
        sound = self._apply_adsr(sound, 0.05, 0.3, 0.6, 0.4, duration)
        
        return self._normalize_audio(sound)
    
    def _apply_adsr(self, sound: np.ndarray, attack: float, decay: float, 
                   sustain_level: float, release: float, total_duration: float) -> np.ndarray:
        """Apply ADSR envelope"""
        n_samples = len(sound)
        envelope = np.ones(n_samples)
        
        # Sample indices
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_samples = n_samples - attack_samples - decay_samples - release_samples
        
        if sustain_samples < 0:
            sustain_samples = 0
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        start_idx = attack_samples
        end_idx = start_idx + decay_samples
        if decay_samples > 0 and end_idx <= n_samples:
            envelope[start_idx:end_idx] = np.linspace(1, sustain_level, decay_samples)
        
        # Sustain phase
        start_idx = attack_samples + decay_samples
        end_idx = start_idx + sustain_samples
        if end_idx <= n_samples:
            envelope[start_idx:end_idx] = sustain_level
        
        # Release phase
        start_idx = n_samples - release_samples
        if start_idx >= 0 and release_samples > 0:
            envelope[start_idx:] = np.linspace(sustain_level, 0, release_samples)
        
        return sound * envelope
    
    def _low_pass_filter(self, sound: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Simple low-pass filter"""
        # Simple exponential moving average
        alpha = cutoff_freq / (cutoff_freq + self.sample_rate)
        filtered = np.zeros_like(sound)
        filtered[0] = sound[0]
        
        for i in range(1, len(sound)):
            filtered[i] = alpha * sound[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    def _normalize_audio(self, sound: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio level"""
        max_val = np.max(np.abs(sound))
        if max_val > 0:
            sound = sound * target_level / max_val
        return sound
    
    def save_wav(self, sound: np.ndarray, filename: str):
        """Save as WAV file"""
        # Convert to 16-bit integer
        audio_data = (sound * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_data)
        print(f"✅ Sound file saved: {filename}")
    
    def create_all_assets(self, base_path: str = "assets/sounds"):
        """Create all sound assets"""
        print("🎵 Creating sound assets...")
        
        # Note frequencies
        notes = {
            'C4': 261.63,
            'D4': 293.66,
            'E4': 329.63,
            'F4': 349.23,
            'G4': 392.00
        }
        
        # Piano sounds
        piano_path = os.path.join(base_path, "piano")
        os.makedirs(piano_path, exist_ok=True)
        
        for note, freq in notes.items():
            sound = self.generate_piano_note(freq, duration=1.2)
            self.save_wav(sound, os.path.join(piano_path, f"{note}.wav"))
        
        # Drum sounds
        drums_path = os.path.join(base_path, "drums")
        os.makedirs(drums_path, exist_ok=True)
        
        drum_types = ["kick", "snare", "hihat", "crash", "ride"]
        for drum in drum_types:
            sound = self.generate_drum_hit(drum, duration=1.0)
            self.save_wav(sound, os.path.join(drums_path, f"{drum}.wav"))
        
        # Synth sounds
        synth_path = os.path.join(base_path, "synth")
        os.makedirs(synth_path, exist_ok=True)
        
        for note, freq in notes.items():
            sound = self.generate_synth_note(freq, "sawtooth", duration=1.0)
            self.save_wav(sound, os.path.join(synth_path, f"synth_{note}.wav"))
        
        print("✅ All sound assets created!")


if __name__ == "__main__":
    generator = SoundGenerator()
    generator.create_all_assets() 