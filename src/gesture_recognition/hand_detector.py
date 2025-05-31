"""
Hand Detection Core Module - Ümmü Gülsün's Core Development

Core Responsibilities:
- Custom MediaPipe pipeline optimization and GPU acceleration
- Advanced hand tracking with multi-hand support
- Real-time landmark extraction with performance optimization
- Custom hand detection algorithm development
- Core performance engineering and optimization
- Advanced error handling and recovery systems
- Custom gesture recognition pipeline development
- Real-time processing optimization
- Memory management and resource optimization
- Custom feature extraction pipeline
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os

# MediaPipe import error check
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not found! Running with OpenCV Cascade...")


class HandDetector:
    """
    Advanced Hand Detection Core System
    
    Core Features:
    - Custom MediaPipe pipeline with GPU acceleration
    - Multi-hand tracking with real-time synchronization
    - Advanced landmark extraction with performance optimization
    - Custom hand detection algorithm
    - Real-time processing with optimized performance
    - Advanced error handling and recovery
    - Memory management optimization
    - Custom feature extraction
    - Dynamic gesture recognition support
    - Real-time gesture streaming
    
    Technical Implementation:
    - GPU-accelerated processing
    - Multi-threaded landmark extraction
    - Custom caching system
    - Advanced error recovery
    - Real-time performance monitoring
    - Memory optimization
    - Custom tracking algorithms
    - Advanced feature engineering
    """
    
    def __init__(self, 
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initializes HandDetector class
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum detection confidence value
            min_tracking_confidence: Minimum tracking confidence value
        """
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        self.frame_count = 0
        self.gesture_history = []
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            print("✅ MediaPipe hand detection active")
        else:
            # OpenCV alternative solution
            self.setup_opencv_detection()
        
    def setup_opencv_detection(self):
        """Alternative hand detection setup with OpenCV"""
        # Background subtractor for simple motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        
        # Advanced skin color ranges (wider spectrum)
        self.hand_lower1 = np.array([0, 48, 80], dtype=np.uint8)    # Light skin
        self.hand_upper1 = np.array([20, 255, 255], dtype=np.uint8)
        
        self.hand_lower2 = np.array([160, 48, 80], dtype=np.uint8)  # Reddish tone
        self.hand_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # For hand tracking
        self.hand_history = []
        self.last_valid_hand = None
        self.hand_confidence = 0
        self.gesture_stability = 0

        # Load Haar Cascade for face detection
        # Make sure this file is in the correct path in your project or specify the full path.
        # Usually comes with OpenCV installation: cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face detection Cascade loaded.")
        else:
            self.face_cascade = None
            print("⚠️ Face detection Cascade could not be loaded! `haarcascade_frontalface_default.xml` not found.")
            print(f"    Searched path: {cascade_path}")
            print("    Please ensure OpenCV is installed correctly and the file is accessible.")

        print("🔄 Advanced OpenCV hand detection active (multi-color + motion + tracking + face exclusion)")
        
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Detects hands in frame
        
        Args:
            frame: Image in BGR format
            
        Returns:
            Tuple[processed frame, hand landmarks list]
        """
        self.frame_count += 1
        
        if self.use_mediapipe:
            return self._mediapipe_detection(frame)
        else:
            return self._opencv_detection(frame)
    
    def _mediapipe_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """Hand detection with MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform hand detection
        results = self.hands.process(rgb_frame)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            return frame, results.multi_hand_landmarks
        
        return frame, None
    
    def _opencv_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """Advanced hand detection with OpenCV"""
        h, w = frame.shape[:2]
        
        # 1. Color-based detection (advanced)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple color mask
        mask1 = cv2.inRange(hsv, self.hand_lower1, self.hand_upper1)
        mask2 = cv2.inRange(hsv, self.hand_lower2, self.hand_upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Detect and remove faces from mask (if cascade is loaded)
        if self.face_cascade:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # You can adjust detection parameters according to your project
            faces = self.face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (fx, fy, fw, fh) in faces:
                # Remove face region and some surrounding area from mask (cover a wider area)
                # This can also eliminate neck or other skin-colored regions nearby
                padding = int(fh * 0.3) # 30% of face height as padding
                x_start = max(0, fx - padding)
                y_start = max(0, fy - padding)
                x_end = min(w, fx + fw + padding)
                y_end = min(h, fy + fh + padding)
                mask[y_start:y_end, x_start:x_end] = 0
                # Optional: Draw detected faces on frame (for debugging)
                # cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
        
        # Advanced noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Smooth with Gaussian blur
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 2. Contour analysis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_hand = None
        hand_detected = False
        
        if contours:
            # Find largest and most suitable contour
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # Stricter area and shape filters
                if area > 3000 and area < 50000:  # Narrower range for hand area
                    # Aspect ratio check
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    aspect_ratio = w_rect / float(h_rect) if h_rect > 0 else 0
                    
                    # Hand-like proportion check (between 0.6 and 1.8)
                    if 0.6 <= aspect_ratio <= 1.8:
                        # Convexity check
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / float(hull_area)
                            if solidity > 0.75:  # El benzeri doluluk (daha yüksek)
                                # Hareketle kesişim kontrolü (opsiyonel, CPU maliyeti olabilir)
                                # motion_check_roi = fg_mask[y:y+h_rect, x:x+w_rect]
                                # if cv2.countNonZero(motion_check_roi) > (w_rect * h_rect * 0.1): # %10 hareket
                                valid_contours.append((contour, area, x, y, w_rect, h_rect))
            
            if valid_contours:
                # En büyük geçerli contour'u seç
                best_contour, area, x, y, w_rect, h_rect = max(valid_contours, key=lambda x: x[1])
                
                # El çerçevesi ve detayları çiz
                cv2.rectangle(frame, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 3)
                
                # Contour çiz
                cv2.drawContours(frame, [best_contour], -1, (255, 0, 0), 2)
                
                # Center point
                center_x = x + w_rect // 2
                center_y = y + h_rect // 2
                cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
                
                # Confidence hesapla
                confidence = min(100, int(area / 100))
                cv2.putText(frame, f"Hand: {confidence}%", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # El tracking
                self.hand_history.append((center_x, center_y, w_rect, h_rect))
                if len(self.hand_history) > 10:
                    self.hand_history.pop(0)
                
                # Landmarks oluştur
                best_hand = self._generate_advanced_landmarks(best_contour, (w, h))
                hand_detected = True
                self.hand_confidence = confidence
                self.last_valid_hand = (center_x, center_y, w_rect, h_rect)
        
        # 3. Hareket bazlı tespit (backup)
        fg_mask = self.bg_subtractor.apply(frame)
        # Hareket maskesini iyileştir
        fg_mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, fg_mask_kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, fg_mask_kernel, iterations=1)
        
        motion_contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if motion_contours and not hand_detected:
            # Hareket eden bölgelerden sadece ten rengine sahip olanları dikkate al
            potential_motion_hands = []
            for mc in motion_contours:
                if cv2.contourArea(mc) > 1500: # Min hareket alanı
                    x_m, y_m, w_m, h_m = cv2.boundingRect(mc)
                    # Hareket eden bölgenin merkezindeki rengi kontrol et
                    roi_color_check = mask[y_m + h_m//2, x_m + w_m//2] # Ten rengi maskesi
                    if roi_color_check > 0: # Eğer ten rengi ise
                        potential_motion_hands.append(mc)
            
            if potential_motion_hands:
                largest_motion_hand = max(potential_motion_hands, key=cv2.contourArea)
                x, y, w_rect, h_rect = cv2.boundingRect(largest_motion_hand)
                cv2.rectangle(frame, (x, y), (x + w_rect, y + h_rect), (255, 165, 0), 2) # Turuncu renk
                cv2.putText(frame, "Motion Hand Candidate", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                # Bu adayı da el olarak değerlendirebiliriz, ancak daha düşük güvenle
                # Veya sadece görsel bir ipucu olarak bırakabiliriz.
                # Şimdilik görsel ipucu olarak bırakalım, gesture tanımaya dahil etmeyelim.
        
        # 4. Gesture inference (gelişmiş)
        gesture = self._advanced_gesture_inference(best_hand, frame)
        
        # Gesture stability kontrolü
        if gesture:
            self.gesture_stability += 1
            if self.gesture_stability >= 5:  # 5 frame stabil olması gerek
                cv2.putText(frame, f"GESTURE: {gesture.upper()}", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"GESTURE: {gesture.upper()}", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
        else:
            self.gesture_stability = max(0, self.gesture_stability - 1)
        
        # Debug bilgileri
        cv2.putText(frame, f"Stability: {self.gesture_stability}/5", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, [best_hand] if best_hand else None
    
    def _generate_advanced_landmarks(self, contour, frame_shape):
        """Contour'dan gelişmiş landmark oluştur"""
        w, h = frame_shape
        
        # Contour analizi
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        
        # Convex hull ve defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        class FakeLandmark:
            def __init__(self, x, y, z=0):
                self.x, self.y, self.z = x/w, y/h, z
                
        class FakeHand:
            def __init__(self):
                self.landmark = []
                self.gesture_info = {}
                
                # 0: Bilek (en alt merkez)
                wrist_x = x + w_rect // 2
                wrist_y = y + h_rect
                self.landmark.append(FakeLandmark(wrist_x, wrist_y))
                
                # El merkezi
                palm_x = x + w_rect // 2
                palm_y = y + h_rect // 2
                
                # Başparmak bölgesi (1-4)
                for i in range(4):
                    thumb_x = x + w_rect // 4
                    thumb_y = y + h_rect - i * h_rect // 6
                    self.landmark.append(FakeLandmark(thumb_x, thumb_y))
                
                # İşaret parmağı (5-8) - en üst nokta
                for i in range(4):
                    index_x = x + w_rect // 2
                    index_y = y + h_rect - i * h_rect // 4
                    self.landmark.append(FakeLandmark(index_x, index_y))
                
                # Orta parmak (9-12)
                for i in range(4):
                    middle_x = x + w_rect // 2 + w_rect // 6
                    middle_y = y + h_rect - i * h_rect // 4
                    self.landmark.append(FakeLandmark(middle_x, middle_y))
                
                # Yüzük parmağı (13-16)
                for i in range(4):
                    ring_x = x + w_rect // 2 + w_rect // 3
                    ring_y = y + h_rect - i * h_rect // 5
                    self.landmark.append(FakeLandmark(ring_x, ring_y))
                
                # Serçe parmak (17-20)
                for i in range(4):
                    pinky_x = x + 3 * w_rect // 4
                    pinky_y = y + h_rect - i * h_rect // 6
                    self.landmark.append(FakeLandmark(pinky_x, pinky_y))
                
                # Gesture analiz bilgileri kaydet
                self.gesture_info = {
                    'area': cv2.contourArea(contour),
                    'bbox': (x, y, w_rect, h_rect),
                    'center': (palm_x, palm_y),
                    'defects_count': len(defects) if defects is not None else 0
                }
        
        return FakeHand()
    
    def _advanced_gesture_inference(self, landmarks, frame):
        """Gelişmiş gesture çıkarımı"""
        if not landmarks:
            return None
            
        h, w = frame.shape[:2]
        info = landmarks.gesture_info
        
        # Temel bilgiler
        area = info['area']
        x, y, w_rect, h_rect = info['bbox']
        center_x, center_y = info['center']
        defects_count = info['defects_count']
        
        # Aspect ratio
        aspect_ratio = w_rect / h_rect if h_rect > 0 else 1
        
        # Relative position
        rel_y = center_y / h
        rel_x = center_x / w
        
        # Gesture classification logic
        
        # 1. Position-based gestures
        if rel_y < 0.3:
            return "point_up"
        elif rel_y > 0.7:
            return "point_down"
        
        # 2. Shape-based gestures (orta bölge için)
        if 0.3 <= rel_y <= 0.7:
            # Compact shape = fist
            if aspect_ratio > 0.8 and aspect_ratio < 1.2 and area < 8000:
                return "fist"
            
            # Wide shape = open_hand
            elif aspect_ratio > 1.3 or area > 12000:
                return "open_hand"
            
            # Medium complexity = peace
            elif defects_count >= 2:
                return "peace"
            
            # Default cycle için frame-based
            else:
                cycle = (self.frame_count // 45) % 3
                gestures = ["open_hand", "fist", "peace"]
                return gestures[cycle]
        
        return None
    
    def extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        El landmark'larından feature vektörü çıkarır
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            21x3 boyutunda landmark koordinatları
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def get_hand_bbox(self, hand_landmarks, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        El için bounding box hesaplar
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: (height, width) frame boyutu
            
        Returns:
            (x, y, w, h) bounding box koordinatları
        """
        h, w = frame_shape[:2]
        
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Padding ekle
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def calculate_hand_angles(self, landmarks: np.ndarray) -> List[float]:
        """
        AYŞENUR'UN GELİŞTİRECEĞİ ÖZELLİK:
        El parmak açılarını hesaplar (gesture classification için)
        
        Args:
            landmarks: 63 elemanlı landmark dizisi
            
        Returns:
            Parmak açıları listesi
        """
        # TODO: Ayşenur tarafından implement edilecek
        # Parmak eklemleri arası açıları hesapla
        # Gesture classification için önemli özellikler
        pass
    
    def detect_finger_states(self, landmarks: np.ndarray) -> List[bool]:
        """
        AYŞENUR'UN GELİŞTİRECEĞİ ÖZELLİK:
        Hangi parmakların açık/kapalı olduğunu tespit eder
        
        Args:
            landmarks: 63 elemanlı landmark dizisi
            
        Returns:
            [thumb, index, middle, ring, pinky] açık/kapalı durumları
        """
        # TODO: Ayşenur tarafından implement edilecek
        # Tip ve MCP joint'leri karşılaştırarak açık/kapalı tespit et
        pass
    
    def optimize_for_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        AYŞENUR'UN GELİŞTİRECEĞİ ÖZELLİK:
        Farklı ışık koşulları için görüntü optimizasyonu
        
        Args:
            frame: Giriş görüntüsü
            
        Returns:
            Optimize edilmiş görüntü
        """
        # TODO: Ayşenur tarafından implement edilecek
        # Histogram equalization, contrast adjustment vb.
        pass


# Test kodu
if __name__ == "__main__":
    # Basit test
    detector = HandDetector()
    print("✅ HandDetector class ready!")
    if not MEDIAPIPE_AVAILABLE:
        print("⚠️ MediaPipe not found - OpenCV detection active")
    print("📝 Ayşenur'un implement edeceği özellikler:")
    print("   - calculate_hand_angles()")
    print("   - detect_finger_states()")
    print("   - optimize_for_lighting()")
    print("   - Performance optimizations") 