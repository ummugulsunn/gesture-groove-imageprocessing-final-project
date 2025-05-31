"""
Sound Manager Module - Ümmu Gülsün's Task
Sound management and playback system using Pygame
"""

import pygame
import numpy as np
import os
from typing import Dict, Optional, List
import threading
import time

# Optional Pydub import
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ Pydub not found! Sound formats will be limited...")


class SoundManager:
    """
    Manages and plays sound files
    
    Features to be developed by Ümmu Gülsün:
    - Low latency sound playback (< 50ms)
    - Multiple instrument support
    - Sound effects (reverb, delay, chorus)
    - MIDI integration
    - Real-time sound synthesis
    """
    
    def __init__(self, 
                 frequency: int = 44100,
                 buffer_size: int = 512,
                 channels: int = 2):
        """
        Initializes SoundManager
        
        Args:
            frequency: Sampling frequency (Hz)
            buffer_size: Audio buffer size
            channels: Number of channels (1=mono, 2=stereo)
        """
        try:
            # Initialize Pygame mixer
            pygame.mixer.pre_init(frequency=frequency, size=-16, channels=channels, buffer=buffer_size)
            pygame.mixer.init()
            
            self.frequency = frequency
            self.buffer_size = buffer_size
            self.channels = channels
            
            # Sound collections
            self.sounds = {}  # Simple sounds
            self.instruments = {}  # Instrument packages
            self.loaded_sounds = {}  # Loaded pygame Sound objects
            
            # Audio state
            self.master_volume = 1.0
            self.is_playing = {}
            
            # For beat pattern
            self.beat_thread = None
            self.beat_running = False
            self.pattern_callback = None
            self.current_bpm = 120  # Default BPM
            
            print(f"🎵 SoundManager initialized - {frequency}Hz, buffer: {buffer_size}")
            
        except pygame.error as e:
            print(f"❌ SoundManager initialization error: {e}")
            raise
        
    def load_sound_file(self, file_path: str, sound_id: str) -> bool:
        """
        Loads and caches sound file
        
        Args:
            file_path: Sound file path
            sound_id: Unique ID for the sound
            
        Returns:
            Successful loading status
        """
        try:
            if os.path.exists(file_path):
                sound = pygame.mixer.Sound(file_path)
                self.sounds[sound_id] = sound
                print(f"✅ Sound loaded: {sound_id}")
                return True
            else:
                print(f"❌ Sound file not found: {file_path}")
                return False
        except Exception as e:
            print(f"❌ Sound loading error: {e}")
            return False
    
    def generate_synthetic_sound(self, 
                                frequency: float, 
                                duration: float, 
                                wave_type: str = 'sine',
                                sound_id: str = None) -> pygame.mixer.Sound:
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Real-time sound synthesis
        
        Args:
            frequency: Sound frequency (Hz)
            duration: Duration (seconds)
            wave_type: Wave type ('sine', 'square', 'sawtooth', 'triangle')
            sound_id: ID for caching
            
        Returns:
            Pygame Sound object
        """
        sample_rate = self.frequency
        frames = int(duration * sample_rate)
        
        # Generate sound based on wave type
        if wave_type == 'sine':
            arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
        elif wave_type == 'square':
            # TO BE IMPLEMENTED BY ÜMMU GÜLSÜN: Square wave generator
            t = np.linspace(0, duration, frames)
            arr = np.sign(np.sin(2 * np.pi * frequency * t))
            # Light smoothing for anti-aliasing
            arr = np.convolve(arr, np.ones(3)/3, mode='same')
        elif wave_type == 'sawtooth':
            # TO BE IMPLEMENTED BY ÜMMU GÜLSÜN: Sawtooth wave generator
            t = np.linspace(0, duration, frames)
            # Sawtooth: linear increase from -1 to 1
            arr = 2 * (t * frequency % 1) - 1
            # Harmonic enrichment
            arr += 0.3 * np.sin(4 * np.pi * frequency * t)
            arr += 0.1 * np.sin(8 * np.pi * frequency * t)
        elif wave_type == 'triangle':
            # TO BE IMPLEMENTED BY ÜMMU GÜLSÜN: Triangle wave generator
            t = np.linspace(0, duration, frames)
            # Triangle wave: absolute value of sawtooth
            sawtooth = 2 * (t * frequency % 1) - 1
            arr = 2 * np.abs(sawtooth) - 1
            # For smoothing
            arr = np.clip(arr, -1, 1)
        else:
            arr = np.zeros(frames)
        
        # Apply envelope (fade in/out)
        fade_frames = frames // 10
        arr[:fade_frames] *= np.linspace(0, 1, fade_frames)
        arr[-fade_frames:] *= np.linspace(1, 0, fade_frames)
        
        # Adjust volume
        arr *= self.master_volume
        
        # Duplicate for stereo
        if self.channels == 2:
            arr = np.array([arr, arr]).T
            arr = np.ascontiguousarray(arr)  # Make C-contiguous
        
        # Convert to Pygame sound object
        arr = (arr * 32767).astype(np.int16)
        
        # Ensure array is contiguous
        arr = np.ascontiguousarray(arr)
        
        sound = pygame.sndarray.make_sound(arr)
        
        # Add to cache
        if sound_id:
            self.sounds[sound_id] = sound
            
        return sound
    
    def load_instrument_pack(self, instrument_name: str) -> bool:
        """
        Loads specified instrument package
        
        Args:
            instrument_name: 'piano', 'drums', 'synth', 'guitar'
            
        Returns:
            bool: Loading successful
        """
        try:
            if instrument_name == 'piano':
                sounds = self._generate_piano_pack()
            elif instrument_name == 'drums':
                sounds = self._generate_drum_pack()
            elif instrument_name == 'synth':
                sounds = self._generate_synth_pack()
            elif instrument_name == 'guitar':
                sounds = self._generate_guitar_pack()
            else:
                print(f"❌ Unknown instrument: {instrument_name}")
                return False
            
            # Save instrument package
            self.instruments[instrument_name] = sounds
            return True
            
        except Exception as e:
            print(f"❌ {instrument_name} loading error: {e}")
            return False
    
    def play_instrument_sound(self, instrument: str, sound_id: str, volume: float = 1.0) -> bool:
        """
        Plays specified instrument sound
        
        Args:
            instrument: Instrument name
            sound_id: Sound ID
            volume: Sound level
        """
        try:
            if instrument in self.instruments:
                if sound_id in self.instruments[instrument]:
                    # Numpy array to pygame Sound
                    sound_array = self.instruments[instrument][sound_id]
                    
                    # Check if it exists in cache
                    cache_key = f"{instrument}_{sound_id}"
                    if cache_key not in self.loaded_sounds:
                        # Float32 array to int16
                        if sound_array.dtype == np.float32:
                            sound_int16 = (sound_array * 32767).astype(np.int16)
                        else:
                            sound_int16 = sound_array.astype(np.int16)
                        
                        # Reshape for stereo
                        if self.channels == 2 and len(sound_int16.shape) == 1:
                            sound_int16 = np.column_stack((sound_int16, sound_int16))
                        
                        # Create pygame Sound
                        pygame_sound = pygame.sndarray.make_sound(sound_int16)
                        self.loaded_sounds[cache_key] = pygame_sound
                    
                    # Play sound
                    sound = self.loaded_sounds[cache_key]
                    sound.set_volume(volume * self.master_volume)
                    sound.play()
                    return True
                else:
                    print(f"❌ Sound not found: {instrument}/{sound_id}")
                    return False
            else:
                print(f"❌ Instrument not found: {instrument}")
                return False
                
        except Exception as e:
            print(f"❌ Sound playback error: {e}")
            return False
    
    def play_note(self, 
                  instrument: str, 
                  note: str, 
                  velocity: float = 1.0) -> bool:
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Plays specified instrument and note
        
        Args:
            instrument: Instrument name
            note: Note (e.g: 'C4', 'A#3')
            velocity: Playback power (0.0-1.0)
            
        Returns:
            Successful playback status
            
        Implementation Guide:
        1. Convert note to MIDI note number
        2. Calculate frequency (A4=440Hz reference)
        3. Apply volume mapping based on velocity
        4. Generate or play sound based on instrument type
        """
        # MIDI note mapping dictionary
        note_to_midi = {
            'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
            'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8,
            'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
        }
        
        try:
            # Note parse (e.g: 'C4' -> note='C', octave=4)
            if len(note) >= 2:
                note_name = note[:-1].upper()
                octave = int(note[-1])
                
                if note_name in note_to_midi:
                    # MIDI note number calculation (C4 = 60)
                    midi_note = note_to_midi[note_name] + (octave + 1) * 12
                    
                    # Frequency calculation (A4=440Hz, MIDI note 69)
                    frequency = 440.0 * (2 ** ((midi_note - 69) / 12))
                    
                    # Volume mapping based on velocity (MIDI style)
                    volume = velocity * self.master_volume
                    
                    # First check if file exists
                    sound_id = f"{note.lower()}"
                    if self.play_instrument_sound(instrument, sound_id, volume):
                        return True
                    
                    # If file doesn't exist, generate synthetic
                    duration = 1.0  # Default note duration
                    
                    # Select wave form based on instrument type
                    wave_type = 'sine'  # Default
                    if instrument.lower() == 'piano':
                        wave_type = 'sine'
                    elif instrument.lower() == 'synth':
                        wave_type = 'sawtooth'
                    elif instrument.lower() == 'organ':
                        wave_type = 'square'
                    
                    # Generate and play sound
                    temp_sound = self.generate_synthetic_sound(
                        frequency=frequency,
                        duration=duration,
                        wave_type=wave_type
                    )
                    
                    temp_sound.set_volume(volume)
                    temp_sound.play()
                    
                    print(f"🎵 {instrument}: {note} ({frequency:.1f}Hz) velocity:{velocity:.2f}")
                    return True
                else:
                    print(f"❌ Invalid note: {note_name}")
                    return False
            else:
                print(f"❌ Invalid note format: {note}")
                return False
                
        except Exception as e:
            print(f"❌ Note playback error: {e}")
            return False
    
    def add_reverb_effect(self, sound: pygame.mixer.Sound, 
                         room_size: float = 0.5,
                         damping: float = 0.5) -> pygame.mixer.Sound:
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Adds reverb effect
        
        Args:
            sound: Sound to apply effect to
            room_size: Room size (0.0-1.0)
            damping: Damping (0.0-1.0)
            
        Returns:
            Effected sound
            
        Implementation Strategy:
        1. Convert sound to numpy array
        2. Create multiple delay lines
        3. Add all-pass filters
        4. Create reverb with comb filters
        5. Mix dry/wet signals
        """
        try:
            # Sound to numpy array
            sound_array = pygame.sndarray.array(sound)
            
            # Stereo control
            if len(sound_array.shape) == 1:
                # Mono -> Stereo
                sound_array = np.array([sound_array, sound_array]).T
            
            # Reverb parameters
            delay_times = [0.03, 0.05, 0.07, 0.11]  # Multiple delays (seconds)
            decay_factor = 1.0 - damping
            room_factor = room_size * 0.8
            
            reverb_signal = np.zeros_like(sound_array, dtype=np.float32)
            original_signal = sound_array.astype(np.float32) / 32767.0
            
            # Calculate reverb for each delay line
            for delay_time in delay_times:
                delay_samples = int(delay_time * self.frequency)
                
                if delay_samples > 0 and delay_samples < len(original_signal):
                    # Create delayed signal
                    delayed = np.zeros_like(original_signal)
                    delayed[delay_samples:] = original_signal[:-delay_samples]
                    
                    # Apply decay
                    decay_envelope = np.exp(-np.arange(len(delayed)) * decay_factor / self.frequency)
                    delayed *= decay_envelope.reshape(-1, 1)
                    
                    # Scale with room size
                    delayed *= room_factor
                    
                    # Add to reverb
                    reverb_signal += delayed
            
            # Mix dry and wet signals
            wet_amount = 0.3
            dry_amount = 1.0 - wet_amount
            
            final_signal = dry_amount * original_signal + wet_amount * reverb_signal
            
            # Normalize and clamp
            final_signal = np.clip(final_signal, -1.0, 1.0)
            final_signal = (final_signal * 32767).astype(np.int16)
            
            # New Sound object
            final_signal = np.ascontiguousarray(final_signal)
            reverb_sound = pygame.sndarray.make_sound(final_signal)
            
            print(f"✅ Reverb effect added: room={room_size:.2f}, damping={damping:.2f}")
            return reverb_sound
            
        except Exception as e:
            print(f"❌ Reverb effect error: {e}")
            return sound  # Original sound returned
    
    def add_delay_effect(self, sound: pygame.mixer.Sound,
                        delay_time: float = 0.3,
                        feedback: float = 0.4,
                        mix: float = 0.3) -> pygame.mixer.Sound:
        """
        FEATURE TO BE DEVELOPED BY ÜMMU GÜLSÜN:
        Adds delay/echo effect
        
        Args:
            sound: Sound to apply effect to
            delay_time: Delay time (seconds)
            feedback: Feedback amount (0.0-0.9)
            mix: Dry/wet mix ratio (0.0-1.0)
            
        Returns:
            Effected sound
            
        Implementation Steps:
        1. Create delay buffer
        2. Echo chain with feedback loop
        3. Mix control for dry/wet balance
        4. Anti-feedback protection
        """
        try:
            # Sound to numpy array
            sound_array = pygame.sndarray.array(sound)
            
            # Stereo control
            if len(sound_array.shape) == 1:
                sound_array = np.array([sound_array, sound_array]).T
            
            # Parameters
            delay_samples = int(delay_time * self.frequency)
            feedback = np.clip(feedback, 0.0, 0.85)  # Anti-feedback protection
            
            # Original signal
            original = sound_array.astype(np.float32) / 32767.0
            
            # Create delay buffer
            delayed_signal = np.zeros_like(original)
            
            # Simple delay implementation
            if delay_samples > 0 and delay_samples < len(original):
                # First delay
                delayed_signal[delay_samples:] = original[:-delay_samples]
                
                # Feedback delays
                current_delay = delayed_signal.copy()
                total_delayed = delayed_signal.copy()
                
                # Multiple echoes
                for echo in range(1, 5):  # 4 echo
                    echo_delay = delay_samples * (echo + 1)
                    if echo_delay < len(original):
                        echo_signal = np.zeros_like(original)
                        echo_signal[echo_delay:] = original[:-echo_delay]
                        
                        # Exponential decay
                        decay = feedback ** echo
                        echo_signal *= decay
                        
                        total_delayed += echo_signal
            
            # Mix dry and wet
            dry_signal = original * (1.0 - mix)
            wet_signal = total_delayed * mix
            
            final_signal = dry_signal + wet_signal
            
            # Normalize
            final_signal = np.clip(final_signal, -1.0, 1.0)
            final_signal = (final_signal * 32767).astype(np.int16)
            
            # Sound object
            final_signal = np.ascontiguousarray(final_signal)
            delay_sound = pygame.sndarray.make_sound(final_signal)
            
            print(f"✅ Delay effect added: time={delay_time:.2f}s, feedback={feedback:.2f}")
            return delay_sound
            
        except Exception as e:
            print(f"❌ Delay effect error: {e}")
            return sound
    
    def create_beat_pattern(self, pattern: List[str], bpm: int = 120, ui_callback=None, instrument: str = None, gesture_sound_mapping: dict = None):
        """
        Beat pattern system - plays gesture pattern with selected instrument sounds
        """
        try:
            self.stop_beat_pattern()
            print(f"🥁 Beat Pattern creating: {pattern} @ {bpm} BPM")
            self.current_bpm = bpm
            self.beat_running = True
            # If instrument and mapping parameters are not provided, use drums and default mapping
            if instrument is None:
                instrument = 'drums'
            if gesture_sound_mapping is None:
                gesture_sound_mapping = {
                    'open_hand': 'kick',
                    'fist': 'snare',
                    'point_up': 'hihat',
                    'point_down': 'crash',
                    'peace': 'ride'
                }
            self.beat_thread = threading.Thread(
                target=self._beat_loop_gesture,
                args=(pattern, bpm, instrument, gesture_sound_mapping),
                daemon=True
            )
            self.beat_thread.start()
            return True
        except Exception as e:
            print(f"❌ Beat pattern error: {e}")
            return False

    def _beat_loop_gesture(self, pattern: List[str], bpm: int, instrument: str, gesture_sound_mapping: dict):
        """Gesture pattern'ı seçili enstrümanın mapping'iyle çalar"""
        try:
            beat_duration = 1.2 # Saniye cinsinden adım arası süre (daha yavaş ve anlaşılır)
            while self.beat_running:
                for i, gesture in enumerate(pattern):
                    if not self.beat_running:
                        break
                    # Callback'i çağır
                    self._call_pattern_callback(i, gesture)
                    # Gesture'ı enstrüman mapping'inden ses ID'sine çevir
                    sound_id = gesture_sound_mapping.get(gesture)
                    if sound_id:
                        self.play_instrument_sound(instrument, sound_id)
                    time.sleep(beat_duration)
        except Exception as e:
            print(f"❌ Beat pattern error: {e}")
        finally:
            self.beat_running = False
            print("🛑 Beat pattern stopped")
    
    def stop_beat_pattern(self):
        """Stops beat pattern"""
        try:
            self.beat_running = False
            if hasattr(self, 'beat_thread') and self.beat_thread and self.beat_thread.is_alive():
                self.beat_thread.join(timeout=1.0)
            print("🛑 Beat pattern stopped")
        except Exception as e:
            print(f"❌ Beat pattern stop error: {e}")
            
    def set_pattern_callback(self, callback):
        """Sets beat pattern callback"""
        self.pattern_callback = callback
        
    def _call_pattern_callback(self, position: int, step: str):
        """Calls pattern callback"""
        if self.pattern_callback and callable(self.pattern_callback):
            try:
                self.pattern_callback(position, step)
            except Exception as e:
                print(f"❌ Pattern callback error: {e}")
                
    def set_master_volume(self, volume: float):
        """Sets master volume"""
        self.master_volume = max(0.0, min(1.0, volume))
        
    def stop_all_sounds(self):
        """Stops all sounds"""
        pygame.mixer.stop()
        
    def get_latency_info(self) -> Dict:
        """
        Returns system information
        """
        return {
            'frequency': self.frequency,
            'buffer_size': self.buffer_size,
            'channels': self.channels,
            'estimated_latency_ms': (self.buffer_size / self.frequency) * 1000,
            'pydub_available': PYDUB_AVAILABLE
        }

    def play_sound(self, sound_id: str, volume: float = 1.0) -> bool:
        """
        Plays sound (backward compatibility)
        
        Args:
            sound_id: Sound ID to play
            volume: Sound level (0.0-1.0)
            
        Returns:
            Successful playback status
        """
        if sound_id in self.sounds:
            sound = self.sounds[sound_id]
            sound.set_volume(volume * self.master_volume)
            sound.play()
            return True
        else:
            print(f"❌ Sound not found: {sound_id}")
            return False

    def _beat_loop(self, pattern: List[str], bpm: int):
        """Beat pattern loop thread"""
        try:
            beat_duration = 60.0 / bpm
            while self.beat_running:
                for i, step in enumerate(pattern):
                    if not self.beat_running:
                        break
                        
                    # Callback'i çağır
                    self._call_pattern_callback(i, step)
                    
                    if step != '0':  # '0' = empty beat
                        if step in ['kick', 'snare', 'hihat', 'crash', 'ride']:
                            self.play_instrument_sound('drums', step)
                        elif 'synth' in step:
                            self.play_instrument_sound('synth', step)
                        else:
                            self.play_instrument_sound('piano', step)
                    
                    time.sleep(beat_duration)
                    
        except Exception as e:
            print(f"❌ Beat pattern error: {e}")
        finally:
            self.beat_running = False
            print("🛑 Beat pattern stopped")

    def _generate_guitar_pack(self) -> Dict[str, np.ndarray]:
        """GERÇEKÇİ Gitar Sesi - Karplus-Strong + Inharmonicity"""
        guitar_sounds = {}
        
        # Single note frequencies
        guitar_notes = {
            'C4': 261.63,
            'D4': 293.66,
            'E4': 329.63,
            'F4': 349.23,
            'G4': 392.00
        }
        
        for note_name, freq in guitar_notes.items():
            # KARPLUS-STRONG INSPIRED ALGORITHM
            N = int(self.frequency / freq)  # Buffer size for Karplus-Strong
            buffer = (np.random.rand(N) - 0.5) * 2  # Initial noise burst (pick)
            
            # Decay factor (how quickly sound dies)
            decay_factor = 0.996  # Slower decay for guitar
            
            # Inharmonicity factor (makes it sound more like a real string)
            inharmonicity = 0.0005 * freq  # Subtle inharmonicity
            
            # Number of samples (2.5 seconds)
            num_samples = int(2.5 * self.frequency)
            guitar_signal = np.zeros(num_samples)
            
            # String simulation loop
            for i in range(num_samples):
                # Karplus-Strong update rule (low-pass filter for decay)
                buffer_val = (buffer[0] + buffer[1]) * 0.5 * decay_factor
                guitar_signal[i] = buffer_val
                
                # Shift buffer (like string vibration)
                buffer = np.roll(buffer, -1)
                buffer[-1] = buffer_val
                
                # Add inharmonicity (subtle pitch drift)
                if i % N == 0 and i > 0:
                    buffer += (np.random.rand(N) - 0.5) * 0.05 * inharmonicity
            
            # ENVELOPE (Guitar specific - pick attack + natural decay)
            attack_time = int(0.005 * self.frequency)  # 5ms very short pick attack
            sustain_level = 0.7
            
            envelope = np.ones_like(guitar_signal)
            
            # Sharp pick attack
            if attack_time > 0:
                envelope[:attack_time] = np.linspace(0, 1, attack_time)
            
            # Natural decay (Karplus-Strong already handles this, but we can shape it)
            decay_start = attack_time
            if decay_start < len(envelope):
                # Exponential decay over the remaining duration
                decay_duration = len(envelope) - decay_start
                decay_curve = np.exp(-np.linspace(0, 3, decay_duration))
                envelope[decay_start:] = sustain_level * decay_curve
            
            # Apply envelope
            guitar_signal *= envelope
            
            # SUBTLE ELECTRIC CHARACTER (Clean guitar, no heavy distortion)
            # Very light compression (pickup character)
            guitar_signal = np.tanh(guitar_signal * 1.1) * 0.9
            
            # Add some brightness (harmonics)
            # This is separate from Karplus-Strong, adds guitar "zing"
            t_harm = np.linspace(0, 2.5, num_samples, False)
            harmonic_zing = 0.1 * np.sin(2 * np.pi * freq * 2 * t_harm) * np.exp(-2 * t_harm)
            harmonic_zing += 0.05 * np.sin(2 * np.pi * freq * 3 * t_harm) * np.exp(-2.5 * t_harm)
            
            # Ensure harmonic_zing is same length and add
            if len(harmonic_zing) == len(guitar_signal):
                guitar_signal += harmonic_zing
            
            # Normalize
            if np.max(np.abs(guitar_signal)) > 0:
                guitar_signal = guitar_signal / np.max(np.abs(guitar_signal)) * 0.6  # Lower volume
            
            guitar_sounds[note_name] = guitar_signal.astype(np.float32)
            
        print(f"🎸 guitar instrument: {len(guitar_sounds)} KARPLUS-STRONG guitar sound generated")
        return guitar_sounds

    def _generate_piano_pack(self) -> Dict[str, np.ndarray]:
        """Piano sound package generation"""
        piano_sounds = {}
        
        # Piano note frequencies (Hz)
        piano_notes = {
            'C4': 261.63,
            'D4': 293.66,
            'E4': 329.63,
            'F4': 349.23,
            'G4': 392.00
        }
        
        for note_name, freq in piano_notes.items():
            t = np.linspace(0, 2.0, int(self.frequency * 2.0), False)  # 2 seconds
            
            # Piano harmonics
            fundamental = 0.6 * np.sin(2 * np.pi * freq * t)
            harmonic2 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            harmonic3 = 0.15 * np.sin(2 * np.pi * freq * 3 * t)
            harmonic4 = 0.1 * np.sin(2 * np.pi * freq * 4 * t)
            
            piano_tone = fundamental + harmonic2 + harmonic3 + harmonic4
            
            # Piano envelope (sharp attack, slow decay)
            attack_time = int(0.05 * self.frequency)  # 50ms attack
            decay_time = int(0.5 * self.frequency)    # 500ms decay
            sustain_level = 0.4
            release_time = int(1.45 * self.frequency) # 1.45s release
            
            envelope = np.ones_like(t)
            
            # Attack
            if attack_time > 0:
                envelope[:attack_time] = np.linspace(0, 1, attack_time)
            
            # Decay
            if decay_time > 0 and attack_time + decay_time < len(envelope):
                envelope[attack_time:attack_time + decay_time] = np.linspace(1, sustain_level, decay_time)
            
            # Sustain
            sustain_end = attack_time + decay_time + int(0.5 * self.frequency)
            if sustain_end < len(envelope):
                envelope[attack_time + decay_time:sustain_end] = sustain_level
            
            # Release
            if sustain_end < len(envelope):
                envelope[sustain_end:] = sustain_level * np.exp(-3 * np.linspace(0, 1, len(envelope) - sustain_end))
            
            piano_sound = piano_tone * envelope
            
            # Normalize
            if np.max(np.abs(piano_sound)) > 0:
                piano_sound = piano_sound / np.max(np.abs(piano_sound)) * 0.7
            
            piano_sounds[note_name] = piano_sound.astype(np.float32)
            
        print(f"🎹 piano instrument: {len(piano_sounds)} sound generated")
        return piano_sounds
    
    def _generate_drum_pack(self) -> Dict[str, np.ndarray]:
        """Drum sound package generation"""
        drum_sounds = {}
        
        # Kick drum
        t_kick = np.linspace(0, 0.5, int(self.frequency * 0.5), False)
        kick_freq = 60  # Low frequency
        kick = 0.8 * np.sin(2 * np.pi * kick_freq * t_kick * np.exp(-3 * t_kick))
        kick_envelope = np.exp(-8 * t_kick)
        kick *= kick_envelope
        drum_sounds['kick'] = kick.astype(np.float32)
        
        # Snare drum
        t_snare = np.linspace(0, 0.3, int(self.frequency * 0.3), False)
        snare_tone = 0.3 * np.sin(2 * np.pi * 200 * t_snare)
        snare_noise = 0.4 * np.random.normal(0, 1, len(t_snare))
        snare = snare_tone + snare_noise
        snare_envelope = np.exp(-12 * t_snare)
        snare *= snare_envelope
        drum_sounds['snare'] = snare.astype(np.float32)
        
        # Hi-hat
        t_hihat = np.linspace(0, 0.1, int(self.frequency * 0.1), False)
        hihat = 0.2 * np.random.normal(0, 1, len(t_hihat))
        hihat_envelope = np.exp(-30 * t_hihat)
        hihat *= hihat_envelope
        drum_sounds['hihat'] = hihat.astype(np.float32)
        
        # Crash
        t_crash = np.linspace(0, 1.0, int(self.frequency * 1.0), False)
        crash = 0.3 * np.random.normal(0, 1, len(t_crash))
        crash_envelope = np.exp(-2 * t_crash)
        crash *= crash_envelope
        drum_sounds['crash'] = crash.astype(np.float32)
        
        # Ride
        t_ride = np.linspace(0, 0.8, int(self.frequency * 0.8), False)
        ride = 0.25 * np.random.normal(0, 1, len(t_ride))
        ride_envelope = np.exp(-1.5 * t_ride)
        ride *= ride_envelope
        drum_sounds['ride'] = ride.astype(np.float32)
        
        print(f"🥁 drums instrument: {len(drum_sounds)} sound generated")
        return drum_sounds
    
    def _generate_synth_pack(self) -> Dict[str, np.ndarray]:
        """Synth sound package generation"""
        synth_sounds = {}
        
        # Synth note frequencies
        synth_notes = {
            'synth_C4': 261.63,
            'synth_D4': 293.66,
            'synth_E4': 329.63,
            'synth_F4': 349.23,
            'synth_G4': 392.00
        }
        
        for note_name, freq in synth_notes.items():
            t = np.linspace(0, 1.5, int(self.frequency * 1.5), False)
            
            # Sawtooth wave for synth
            sawtooth = 2 * (t * freq - np.floor(t * freq + 0.5))
            
            # Low-pass filter effect
            filter_freq = freq * 4
            filter_env = np.exp(-2 * t)
            filtered_sawtooth = sawtooth * filter_env
            
            # ADSR envelope
            attack_time = int(0.02 * self.frequency)  # 20ms
            decay_time = int(0.1 * self.frequency)    # 100ms
            sustain_level = 0.6
            release_time = int(1.38 * self.frequency) # Rest of the time
            
            envelope = np.ones_like(t)
            
            # Attack
            if attack_time > 0:
                envelope[:attack_time] = np.linspace(0, 1, attack_time)
            
            # Decay
            if decay_time > 0 and attack_time + decay_time < len(envelope):
                envelope[attack_time:attack_time + decay_time] = np.linspace(1, sustain_level, decay_time)
            
            # Sustain
            sustain_end = attack_time + decay_time + int(0.3 * self.frequency)
            if sustain_end < len(envelope):
                envelope[attack_time + decay_time:sustain_end] = sustain_level
            
            # Release
            if sustain_end < len(envelope):
                envelope[sustain_end:] = sustain_level * np.exp(-4 * np.linspace(0, 1, len(envelope) - sustain_end))
            
            synth_sound = filtered_sawtooth * envelope * 0.3  # Lower volume for synth
            
            # Normalize
            if np.max(np.abs(synth_sound)) > 0:
                synth_sound = synth_sound / np.max(np.abs(synth_sound)) * 0.6
            
            synth_sounds[note_name] = synth_sound.astype(np.float32)
            
        print(f"🎛️ synth instrument: {len(synth_sounds)} sound generated")
        return synth_sounds


# Test code
if __name__ == "__main__":
    # Simple test
    manager = SoundManager()
    
    # Test sound generation
    test_sound = manager.generate_synthetic_sound(
        frequency=440,  # A note
        duration=0.5,
        wave_type='sine',
        sound_id='test_A'
    )
    
    print("✅ SoundManager class ready!")
    print("📝 Ümmu Gülsün's features to be implemented:")
    print("   - load_instrument_pack()")
    print("   - play_note()")
    print("   - add_reverb_effect()")
    print("   - add_delay_effect()")
    print("   - create_beat_pattern()")
    print("   - Sawtooth/triangle wave synthesis")
    
    # Latency information
    print(f"📊 System information: {manager.get_latency_info()}")
    
    # Test sound playback
    print("🎵 Test sound playing...")
    manager.play_instrument_sound('piano', 'C4')
    
    import time
    time.sleep(1)
    print("✅ Test completed!") 