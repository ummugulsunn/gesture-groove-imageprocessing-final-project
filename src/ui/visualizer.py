"""
Gesture Visualizer Module - Sueda's Task
Real-time gesture visualization and UI dashboard
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Dict, Optional
import time
from collections import deque


class GestureVisualizer:
    """
    Gesture visualization and dashboard system
    
    Features to be developed by Sueda:
    - Real-time gesture graphs
    - Beat Challenge visual feedback
    - Performance dashboard
    - Animated gesture indicators
    - Score and statistics charts
    """
    
    def __init__(self, parent_frame: tk.Frame):
        """
        Initializes GestureVisualizer
        
        Args:
            parent_frame: Main Tkinter frame
        """
        self.parent_frame = parent_frame
        
        # Data storage
        self.gesture_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        self.fps_history = deque(maxlen=50)
        self.score_history = deque(maxlen=50)
        
        # UI components
        self.gesture_indicators = {}
        self.real_time_chart = None
        self.performance_chart = None
        
        # Animation
        self.animation_active = False
        self.current_gesture = None
        
        self.setup_visualizer()
        
    def setup_visualizer(self):
        """Creates visualization panel"""
        # Main container
        main_viz_frame = tk.Frame(self.parent_frame, bg='#34495e')
        main_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Gesture indicators
        self.setup_gesture_indicators(main_viz_frame)
        
        # Right panel - Charts
        self.setup_charts(main_viz_frame)
        
        print("📊 GestureVisualizer ready!")
        
    def setup_gesture_indicators(self, parent):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Creates gesture indicators
        """
        # Left panel
        left_panel = tk.Frame(parent, bg='#34495e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Title
        title = tk.Label(left_panel, 
                        text="🤏 Hand Gestures",
                        font=('Arial', 14, 'bold'),
                        bg='#34495e', fg='white')
        title.pack(pady=(0, 20))
        
        # Gesture indicators
        gestures = [
            ('open_hand', '✋', 'Open Hand', '#3498db'),
            ('fist', '✊', 'Fist', '#e74c3c'),
            ('point_up', '☝️', 'Point Up', '#2ecc71'),
            ('point_down', '👇', 'Point Down', '#f39c12'),
            ('peace', '✌️', 'Peace', '#9b59b6')
        ]
        
        for gesture_id, emoji, name, color in gestures:
            # Gesture container
            gesture_frame = tk.Frame(left_panel, bg='#2c3e50', relief=tk.RAISED, bd=2)
            gesture_frame.pack(fill=tk.X, pady=5, padx=10)
            
            # Emoji
            emoji_label = tk.Label(gesture_frame,
                                 text=emoji,
                                 font=('Arial', 24),
                                 bg='#2c3e50', fg='white')
            emoji_label.pack(side=tk.LEFT, padx=(10, 5), pady=10)
            
            # Name
            name_label = tk.Label(gesture_frame,
                                text=name,
                                font=('Arial', 12, 'bold'),
                                bg='#2c3e50', fg='white')
            name_label.pack(side=tk.LEFT, padx=5, pady=10)
            
            # Confidence bar
            confidence_frame = tk.Frame(gesture_frame, bg='#2c3e50')
            confidence_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10, pady=15)
            
            confidence_bar = ttk.Progressbar(confidence_frame,
                                           length=100,
                                           mode='determinate')
            confidence_bar.pack(fill=tk.X)
            
            # Store references
            self.gesture_indicators[gesture_id] = {
                'frame': gesture_frame,
                'emoji': emoji_label,
                'name': name_label,
                'confidence': confidence_bar,
                'color': color,
                'active': False
            }
            
    def setup_charts(self, parent):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Creates chart panel
        """
        # Right panel
        right_panel = tk.Frame(parent, bg='#34495e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top chart - Real-time gesture
        self.setup_realtime_chart(right_panel)
        
        # Bottom chart - Performance
        self.setup_performance_chart(right_panel)
        
    def setup_realtime_chart(self, parent):
        """Real-time gesture chart"""
        # Chart frame
        chart_frame = tk.Frame(parent, bg='#34495e')
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Matplotlib figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 3), facecolor='#34495e')
        ax.set_facecolor('#2c3e50')
        
        # Chart title
        ax.set_title('Real-Time Gesture Confidence', color='white', fontsize=12)
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Confidence (%)', color='white')
        ax.tick_params(colors='white')
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.real_time_chart = {'fig': fig, 'ax': ax, 'canvas': canvas}
        
    def setup_performance_chart(self, parent):
        """Performance chart"""
        # Chart frame
        chart_frame = tk.Frame(parent, bg='#34495e')
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), facecolor='#34495e')
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
        
        # FPS chart
        ax1.set_title('FPS', color='white', fontsize=10)
        ax1.set_ylabel('FPS', color='white')
        
        # Score chart
        ax2.set_title('Score', color='white', fontsize=10)
        ax2.set_ylabel('Points', color='white')
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.performance_chart = {'fig': fig, 'ax1': ax1, 'ax2': ax2, 'canvas': canvas}
        
    def update_gesture_display(self, gesture: str, confidence: float = 1.0):
        """
        FEATURE TO BE DEVELOPED BY SUEDA:
        Updates gesture display with animations
        """
        try:
            # Update gesture history
            self.gesture_history.append(gesture)
            self.confidence_history.append(confidence)
            
            # Update all indicators
            for gesture_id, indicator in self.gesture_indicators.items():
                if gesture_id == gesture:
                    # Active gesture
                    indicator['confidence'].configure(value=confidence * 100)
                    indicator['frame'].configure(bg='#2ecc71')  # Green highlight
                    indicator['active'] = True
                    
                    # Start animation
                    self._animate_gesture_indicator(gesture_id)
                else:
                    # Inactive gesture
                    indicator['confidence'].configure(value=0)
                    indicator['frame'].configure(bg='#2c3e50')
                    indicator['active'] = False
            
            # Update charts
            self.update_realtime_chart()
            
        except Exception as e:
            print(f"❌ Gesture display update error: {e}")
            
    def _animate_gesture_indicator(self, gesture: str):
        """Animates gesture indicator with pulse effect"""
        if not self.animation_active:
            return
            
        def pulse():
            if gesture in self.gesture_indicators:
                indicator = self.gesture_indicators[gesture]
                if indicator['active']:
                    # Pulse animation
                    current_bg = indicator['frame'].cget('bg')
                    if current_bg == '#2ecc71':
                        indicator['frame'].configure(bg='#27ae60')
                    else:
                        indicator['frame'].configure(bg='#2ecc71')
                        
                    # Continue animation
                    self.parent_frame.after(500, pulse)
                    
        pulse()
        
    def update_realtime_chart(self):
        """Updates real-time gesture confidence chart"""
        try:
            if not self.real_time_chart:
                return
                
            ax = self.real_time_chart['ax']
            ax.clear()
            
            # Plot confidence history
            if len(self.confidence_history) > 0:
                x = range(len(self.confidence_history))
                y = [c * 100 for c in self.confidence_history]  # Convert to percentage
                
                ax.plot(x, y, color='#3498db', linewidth=2)
                ax.fill_between(x, y, alpha=0.2, color='#3498db')
                
            # Update chart appearance
            ax.set_facecolor('#2c3e50')
            ax.set_title('Real-Time Gesture Confidence', color='white', fontsize=12)
            ax.set_xlabel('Time', color='white')
            ax.set_ylabel('Confidence (%)', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2)
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            
            # Redraw
            self.real_time_chart['canvas'].draw()
            
        except Exception as e:
            print(f"❌ Real-time chart update error: {e}")
            
    def update_performance_charts(self, fps: float, score: int):
        """Updates performance charts (FPS and Score)"""
        try:
            if not self.performance_chart:
                return
                
            # Update history
            self.fps_history.append(fps)
            self.score_history.append(score)
            
            # FPS Chart
            ax1 = self.performance_chart['ax1']
            ax1.clear()
            
            if len(self.fps_history) > 0:
                x = range(len(self.fps_history))
                ax1.plot(x, self.fps_history, color='#e74c3c', linewidth=2)
                ax1.fill_between(x, self.fps_history, alpha=0.2, color='#e74c3c')
                
            ax1.set_facecolor('#2c3e50')
            ax1.set_title('FPS', color='white', fontsize=10)
            ax1.set_ylabel('FPS', color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.2)
            
            # Score Chart
            ax2 = self.performance_chart['ax2']
            ax2.clear()
            
            if len(self.score_history) > 0:
                x = range(len(self.score_history))
                ax2.plot(x, self.score_history, color='#2ecc71', linewidth=2)
                ax2.fill_between(x, self.score_history, alpha=0.2, color='#2ecc71')
                
            ax2.set_facecolor('#2c3e50')
            ax2.set_title('Score', color='white', fontsize=10)
            ax2.set_ylabel('Points', color='white')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.2)
            
            # Redraw
            self.performance_chart['canvas'].draw()
            
        except Exception as e:
            print(f"❌ Performance charts update error: {e}")
            
    def show_beat_pattern(self, pattern: List[str], current_index: int = 0):
        """Shows beat pattern in visualization"""
        try:
            # TODO: Implement beat pattern visualization
            pass
        except Exception as e:
            print(f"❌ Beat pattern visualization error: {e}")
            
    def create_gesture_heatmap(self) -> tk.Toplevel:
        """Creates gesture heatmap window"""
        try:
            # TODO: Implement gesture heatmap
            pass
        except Exception as e:
            print(f"❌ Heatmap creation error: {e}")
            
    def export_session_report(self) -> str:
        """Exports session statistics report"""
        try:
            # TODO: Implement session report export
            pass
        except Exception as e:
            print(f"❌ Session report export error: {e}")
            
    def reset_visualizer(self):
        """Resets all visualizer data and displays"""
        try:
            # Clear history
            self.gesture_history.clear()
            self.confidence_history.clear()
            self.fps_history.clear()
            self.score_history.clear()
            
            # Reset indicators
            for indicator in self.gesture_indicators.values():
                indicator['confidence'].configure(value=0)
                indicator['frame'].configure(bg='#2c3e50')
                indicator['active'] = False
                
            # Clear charts
            if self.real_time_chart:
                ax = self.real_time_chart['ax']
                ax.clear()
                ax.set_facecolor('#2c3e50')
                self.real_time_chart['canvas'].draw()
                
            if self.performance_chart:
                ax1 = self.performance_chart['ax1']
                ax2 = self.performance_chart['ax2']
                ax1.clear()
                ax2.clear()
                ax1.set_facecolor('#2c3e50')
                ax2.set_facecolor('#2c3e50')
                self.performance_chart['canvas'].draw()
                
        except Exception as e:
            print(f"❌ Visualizer reset error: {e}")
            
    def test_update():
        """Test function for visualizer updates"""
        try:
            # Create test window
            root = tk.Tk()
            root.title("Gesture Visualizer Test")
            root.geometry("800x600")
            
            # Create visualizer
            viz = GestureVisualizer(root)
            
            # Test update function
            def update_test():
                import random
                gestures = ['open_hand', 'fist', 'point_up', 'point_down', 'peace']
                gesture = random.choice(gestures)
                confidence = random.random()
                
                viz.update_gesture_display(gesture, confidence)
                viz.update_performance_charts(random.uniform(20, 30), random.randint(0, 100))
                
                root.after(1000, update_test)
                
            # Start test updates
            update_test()
            
            # Run test window
            root.mainloop()
            
        except Exception as e:
            print(f"❌ Visualizer test error: {e}")

# Test code
if __name__ == "__main__":
    GestureVisualizer.test_update() 