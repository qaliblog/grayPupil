import cv2
import numpy as np
import time
from collections import deque

# Start video capture with optimized settings
cap = cv2.VideoCapture(0)

# Optimize camera settings for faster frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 180)  # Higher FPS for faster detection
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency

# Load Haar cascade for initial face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Haar cascade detection timing (adaptive based on position changes)
haar_detection_interval = 0.5  # Base interval
last_haar_detection_time = 0
haar_face_position = None  # Store the last Haar cascade face position
haar_position_change_threshold = 20  # Pixels threshold for position change detection
haar_last_position = None  # Track last position for change detection

# Size tracking for smart resizing
average_face_size = None  # Track average face size (dynamic exponential moving average)

# Haar cascade position and size averaging (prevents sudden jumps)
average_haar_x = None  # Average x position
average_haar_y = None  # Average y position
average_haar_w = None  # Average width
average_haar_h = None  # Average height
haar_avg_alpha = 0.3  # 30% new, 70% old average (smooth adaptation)

# Movement tracking for angle calculation
previous_haar_center_x = None  # Previous Haar cascade center position
previous_haar_center_y = None
previous_face_center_x = None  # Previous face center position
previous_face_center_y = None

# Heatmap tracking for squares inside Haar cascade face
heatmap = None  # Will store the heatmap
heatmap_haar_position = None  # Store Haar cascade position for heatmap reference
heatmap_reset_distance = 100  # Reset heatmap if Haar cascade moves this far

# Object tracking for consistent detection (optimized for speed)
object_history = deque(maxlen=150)  # Reduced from 300 to 150 frames for speed
min_consistency_frames = 10  # Reduced from 20 to 10 frames for faster consistency
current_objects = []
last_consistent_face = None
consistency_reset_counter = 0
max_reset_frames = 40  # Reduced from 60 to 40 frames for faster reset

# Eye tracking for consistent detection (optimized for speed)
eye_history = deque(maxlen=50)  # Reduced from 100 to 50 frames for speed
min_eye_consistency_frames = 3  # Reduced from 5 to 3 frames for faster eye consistency
consistent_eyes = []  # Store the 2 most consistent eyes
last_consistent_eyes = None

# Eye detection performance tracking
eye_fps_counter = 0
eye_fps_start_time = time.time()
eye_fps = 0

# Display settings
show_enhanced_face = True

print("Hybrid Face & Eye Detection App Started")
print("Press 'q' to quit")
print("Press 'e' to toggle enhanced face display")
print("Features: Haar cascade + Contrast-based detection")
print("Performance: 480x360 resolution, 0.5s Haar interval")

def detect_face_haar(frame):
    """
    Detect faces using Haar cascade with improved tilt detection
    Returns the largest face detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try multiple scale factors and min neighbors for better tilt detection
    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    
    if len(faces) == 0:
        # Try with different parameters for tilted faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(25, 25))
    
    if len(faces) == 0:
        # Try with even more sensitive parameters
        faces = face_cascade.detectMultiScale(gray, 1.15, 1, minSize=(20, 20))
    
    if len(faces) > 0:
        # Return the largest face (most likely to be the main subject)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        return largest_face
    return None

def enhance_contrast(image):
    """
    Enhance contrast of an image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def add_face_padding(face_rect, frame_shape, padding=20):
    """
    Add padding to face rectangle to capture a larger area
    Returns padded face coordinates
    """
    x, y, w, h = face_rect
    frame_height, frame_width = frame_shape[:2]
    
    # Add padding to all sides
    padded_x = max(0, x - padding)
    padded_y = max(0, y - padding)
    padded_w = min(frame_width - padded_x, w + 2 * padding)
    padded_h = min(frame_height - padded_y, h + 2 * padding)
    
    return [padded_x, padded_y, padded_w, padded_h]

def detect_eyes_in_face_contrast(face_roi):
    """
    Detect eyes in a face region using contrast-based method (no Haar cascades)
    Returns list of eye rectangles with adaptive thresholding for daylight
    """
    # Enhance contrast of the face region
    enhanced_face = enhance_contrast(face_roi)
    
    # Convert to grayscale
    gray_face = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2GRAY)
    
    # Apply additional contrast enhancement for eye detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray_face)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    
    # Calculate adaptive threshold parameters based on image brightness
    mean_brightness = np.mean(blurred)
    
    # Adjust threshold parameters based on lighting conditions
    if mean_brightness > 150:  # Bright daylight
        block_size = 15
        c_value = 5
        clip_limit = 4.0
        lighting_mode = "BRIGHT DAYLIGHT"
    elif mean_brightness > 100:  # Normal lighting
        block_size = 11
        c_value = 2
        clip_limit = 3.0
        lighting_mode = "NORMAL LIGHTING"
    else:  # Low light
        block_size = 7
        c_value = 1
        clip_limit = 2.0
        lighting_mode = "LOW LIGHT"
    
    # Apply adaptive thresholding with dynamic parameters
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, block_size, c_value)
    
    # Additional adaptive contrast enhancement for daylight
    if mean_brightness > 120:
        # Apply additional CLAHE for bright conditions
        clahe_bright = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(6, 6))
        thresh = clahe_bright.apply(thresh)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find potential eyes with adaptive parameters
    potential_eyes = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Adaptive size filtering based on lighting conditions
        if mean_brightness > 150:  # Bright daylight - larger eye regions
            min_size = 10
            max_size = 80
            min_area = 40
            max_area = 2000
        elif mean_brightness > 100:  # Normal lighting
            min_size = 8
            max_size = 60
            min_area = 30
            max_area = 1500
        else:  # Low light - smaller, more precise regions
            min_size = 6
            max_size = 50
            min_area = 20
            max_area = 1200
        
        # Filter by size (eyes should be small contours)
        if min_size < w < max_size and min_size < h < max_size:
            # Filter by aspect ratio (eyes are roughly oval)
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 2.0:
                # Filter by position (eyes should be in upper half of face)
                if y < face_roi.shape[0] * 0.6:
                    # Calculate contour area and filter by reasonable size
                    area = cv2.contourArea(contour)
                    if min_area < area < max_area:
                        potential_eyes.append([x, y, w, h])
    
    # Remove overlapping detections
    filtered_eyes = []
    for eye in potential_eyes:
        x1, y1, w1, h1 = eye
        is_unique = True
        
        for other_eye in filtered_eyes:
            x2, y2, w2, h2 = other_eye
            
            # Calculate overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            if overlap_area > 0.5 * min(w1 * h1, w2 * h2):
                is_unique = False
                break
        
        if is_unique:
            filtered_eyes.append(eye)
    
    return filtered_eyes, enhanced_face, lighting_mode, mean_brightness

def draw_eyes_with_enhancement(frame, face_rect, eyes, enhanced_face_roi, show_enhanced=True):
    """
    Draw detected eyes and show contrast enhancement
    """
    x, y, w, h = face_rect
    
    # Draw eyes on the original frame
    for (ex, ey, ew, eh) in eyes:
        # Convert eye coordinates from face ROI to full frame coordinates
        eye_x = x + ex
        eye_y = y + ey
        
        # Draw eye rectangle
        cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (255, 0, 0), 2)
        cv2.putText(frame, 'Eye', (eye_x, eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw eye center point
        eye_center_x = eye_x + ew // 2
        eye_center_y = eye_y + eh // 2
        cv2.circle(frame, (eye_center_x, eye_center_y), 2, (0, 255, 255), -1)
    
    # Create a small window to show the enhanced face region
    if enhanced_face_roi is not None and show_enhanced:
        # Resize enhanced face for display
        display_size = (120, 120)
        enhanced_display = cv2.resize(enhanced_face_roi, display_size)
        
        # Create border around the enhanced face display
        border_size = 2
        enhanced_with_border = cv2.copyMakeBorder(
            enhanced_display, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, value=(0, 255, 0)
        )
        
        # Overlay the enhanced face in the top-right corner
        overlay_x = frame.shape[1] - enhanced_with_border.shape[1] - 10
        overlay_y = 10
        
        # Ensure the overlay fits within the frame
        if overlay_x >= 0 and overlay_y >= 0:
            frame[overlay_y:overlay_y + enhanced_with_border.shape[0], 
                  overlay_x:overlay_x + enhanced_with_border.shape[1]] = enhanced_with_border
            
            # Add label for enhanced face
            cv2.putText(frame, 'Enhanced Face', (overlay_x, overlay_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add contrast enhancement indicator
            cv2.putText(frame, 'CLAHE Applied', (overlay_x, overlay_y + enhanced_with_border.shape[0] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Add eye detection status
            eye_status = f'Eyes Found: {len(eyes)}'
            cv2.putText(frame, eye_status, (overlay_x, overlay_y + enhanced_with_border.shape[0] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0) if len(eyes) > 0 else (0, 0, 255), 1)

def calculate_distance(rect1, rect2):
    """Calculate center distance between two rectangles"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    center1_x = x1 + w1 // 2
    center1_y = y1 + h1 // 2
    center2_x = x2 + w2 // 2
    center2_y = y2 + h2 // 2
    
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

def calculate_overlap(rect1, rect2):
    """Calculate overlap between two rectangles"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Calculate intersection
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left < right and top < bottom:
        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    return 0

def find_most_consistent_object():
    """Find the object that appears most consistently in recent frames"""
    if len(object_history) < min_consistency_frames:
        return None
    
    # Count appearances for each object with position tolerance
    object_counts = {}
    
    for frame_objects in object_history:
        for obj in frame_objects:
            # Create a position-based key (rounded to reduce exact position dependency)
            x, y, w, h = obj
            # Round position to create tolerance for slight movements
            pos_key = (round(x/10)*10, round(y/10)*10, round(w/10)*10, round(h/10)*10)
            object_counts[pos_key] = object_counts.get(pos_key, 0) + 1
    
    # Find object with highest count
    if object_counts:
        most_consistent = max(object_counts.items(), key=lambda x: x[1])
        if most_consistent[1] >= min_consistency_frames:
            # Return the most recent instance of this object
            for frame_objects in reversed(list(object_history)):
                for obj in frame_objects:
                    x, y, w, h = obj
                    pos_key = (round(x/10)*10, round(y/10)*10, round(w/10)*10, round(h/10)*10)
                    if pos_key == most_consistent[0]:
                        return obj
    return None

def update_consistent_face_position(consistent_face, current_objects, most_consistent_object):
    """Update consistent face position and size based on most consistent detected object with distance-based speed"""
    global average_face_size, size_history, average_haar_x, average_haar_y, previous_haar_center_x, previous_haar_center_y, previous_face_center_x, previous_face_center_y
    
    if not consistent_face or not current_objects or not most_consistent_object:
        return consistent_face
    
    # Find the most consistent object in current frame (closest match)
    best_match = None
    best_distance = float('inf')
    
    for obj in current_objects:
        # Calculate distance to the most consistent object
        distance = calculate_distance(most_consistent_object, obj)
        if distance < best_distance:
            best_distance = distance
            best_match = obj
    
    if not best_match:
        return consistent_face
    
    # Get target object for position update
    target_x, target_y, target_w, target_h = best_match
    
    # Calculate distance from current face center to target center
    x, y, w, h = consistent_face
    current_center_x = x + w // 2
    current_center_y = y + h // 2
    target_center_x = target_x + target_w // 2
    target_center_y = target_y + target_h // 2
    
    distance = np.sqrt((current_center_x - target_center_x)**2 + (current_center_y - target_center_y)**2)
    
    # Check if both heatmap average and Haar cascade are moving toward the same position
    both_moving_toward_target = False
    movement_angle = 0
    if average_haar_x is not None and average_haar_y is not None:
        # Calculate Haar cascade average center
        haar_avg_center_x = average_haar_x + (average_haar_w or 0) // 2
        haar_avg_center_y = average_haar_y + (average_haar_h or 0) // 2
        
        # Calculate distances
        haar_to_target = np.sqrt((haar_avg_center_x - target_center_x)**2 + (haar_avg_center_y - target_center_y)**2)
        current_to_target = distance
        
        # Check if both are moving toward the same target (within 30px tolerance)
        if abs(haar_to_target - current_to_target) < 30:
            both_moving_toward_target = True
            
                                 # Calculate movement angle between heatmap center movement and Haar cascade movement
     
     # Calculate actual movement vectors from previous to current positions
    if (previous_haar_center_x is not None and previous_haar_center_y is not None and 
        previous_face_center_x is not None and previous_face_center_y is not None):
        
        # Haar cascade movement vector (from previous to current position)
        haar_movement_dx = haar_avg_center_x - previous_haar_center_x
        haar_movement_dy = haar_avg_center_y - previous_haar_center_y
        
        # Face movement vector (from previous to current position)
        face_movement_dx = current_center_x - previous_face_center_x
        face_movement_dy = current_center_y - previous_face_center_y
        
        # Calculate angle between the two movement vectors
        if (haar_movement_dx != 0 or haar_movement_dy != 0) and (face_movement_dx != 0 or face_movement_dy != 0):
            # Normalize vectors
            haar_magnitude = np.sqrt(haar_movement_dx*haar_movement_dx + haar_movement_dy*haar_movement_dy)
            face_magnitude = np.sqrt(face_movement_dx*face_movement_dx + face_movement_dy*face_movement_dy)
            
            if haar_magnitude > 0 and face_magnitude > 0:
                # Dot product
                dot_product = (haar_movement_dx * face_movement_dx + haar_movement_dy * face_movement_dy) / (haar_magnitude * face_magnitude)
                # Clamp to valid range for arccos
                dot_product = max(-1.0, min(1.0, dot_product))
                # Calculate angle in degrees
                movement_angle = np.arccos(dot_product) * 180 / np.pi
        else:
            movement_angle = 0
    else:
        # Fallback: use simplified approach for first frame
        movement_angle = 0
    
    # Movement speed based on angle alignment - distance doesn't matter when aligned
    if both_moving_toward_target:  # Both moving toward target
        # Ultra-fast angle-based speed: closer to 0 degrees = extremely fast movement
        if movement_angle < 5:  # Perfect alignment (0-5 degrees)
            pos_alpha = 0.001  # 99.9% new position (ultra maximum speed - under 0.1s)
        elif movement_angle < 10:  # Very straight movement (5-10 degrees)
            pos_alpha = 0.005  # 99.5% new position (extremely fast)
        elif movement_angle < 15:  # Straight movement (10-15 degrees)
            pos_alpha = 0.01  # 99% new position (very fast)
        elif movement_angle < 25:  # Moderately straight (15-25 degrees)
            pos_alpha = 0.02  # 98% new position (fast)
        elif movement_angle < 40:  # Somewhat aligned (25-40 degrees)
            pos_alpha = 0.05  # 95% new position (medium fast)
        else:  # Less aligned movement (40+ degrees)
            pos_alpha = 0.1  # 90% new position (normal coordinated speed)
    else:
        # Only use distance-based speed when NOT coordinated
        if distance < 10:  # Very close - maximum speed
            pos_alpha = 0.02  # 98% new position (very fast)
        elif distance < 25:  # Close - high speed
            pos_alpha = 0.1  # 90% new position (fast)
        elif distance < 50:  # Medium close - medium speed
            pos_alpha = 0.25  # 75% new position (medium)
        elif distance < 80:  # Medium distance - slow speed
            pos_alpha = 0.5  # 50% new position (slow)
        elif distance < 120:  # Far - very slow speed
            pos_alpha = 0.8  # 20% new position (very slow)
        else:  # Very far - minimal movement
            pos_alpha = 0.95  # 5% new position (minimal)
    
    # Smart resizing logic with distance-based speed control
    current_size = w * h
    target_size = target_w * target_h
    
    # Update average size dynamically (weighted average)
    if average_face_size is not None:
        # Use exponential moving average for smoother updates
        alpha_avg = 0.1  # 10% new, 90% old average (slow adaptation)
        average_face_size = alpha_avg * current_size + (1 - alpha_avg) * average_face_size
    else:
        average_face_size = current_size
    
    # Calculate size ratio
    size_ratio = current_size / average_face_size
    target_size_ratio = target_size / average_face_size
    
    # Distance-based resizing speed (prevent false size changes when far)
    if distance < 20:  # Very close - normal resizing speed
        base_size_alpha = 0.7  # Normal resizing speed
    elif distance < 50:  # Close - slightly slower resizing
        base_size_alpha = 0.8  # 80% retention (slower)
    elif distance < 100:  # Medium distance - slow resizing
        base_size_alpha = 0.9  # 90% retention (slow)
    else:  # Far - very slow resizing (prevent false size changes)
        base_size_alpha = 0.95  # 95% retention (very slow)
    
    # Smart resizing with distance-based speed control
    if size_ratio < 0.8:  # Too small - very slow resize
        size_alpha = 0.95  # Keep 95% of current size
    elif size_ratio < 0.9:  # Slightly small - slow resize
        size_alpha = 0.9  # Keep 90% of current size
    elif target_size_ratio > 1.2 and distance > 50:  # Getting bigger while far - very slow resize (false detection)
        size_alpha = 0.98  # Keep 98% of current size (prevent false size increase)
    elif size_ratio > 1.3:  # Too big - resize slowly back to average
        size_alpha = 0.8  # Keep 80% of current size, slowly reduce
    else:  # Good size range - use distance-based speed
        size_alpha = base_size_alpha
    
    # Apply position and size updates
    updated_x = int(pos_alpha * x + (1 - pos_alpha) * target_x)
    updated_y = int(pos_alpha * y + (1 - pos_alpha) * target_y)
    updated_w = int(size_alpha * w + (1 - size_alpha) * target_w)
    updated_h = int(size_alpha * h + (1 - size_alpha) * target_h)
    
    # Update previous positions for movement tracking
    previous_haar_center_x = haar_avg_center_x
    previous_haar_center_y = haar_avg_center_y
    previous_face_center_x = current_center_x
    previous_face_center_y = current_center_y
    
    return [updated_x, updated_y, updated_w, updated_h]

def should_reset_consistency():
    """Check if consistency should be reset due to no face detection"""
    global consistency_reset_counter
    
    if len(current_objects) == 0:
        consistency_reset_counter += 1
    else:
        consistency_reset_counter = 0
    
    return consistency_reset_counter >= max_reset_frames

def find_most_consistent_eyes():
    """Find the 2 most consistent eyes in recent frames"""
    if len(eye_history) < min_eye_consistency_frames:
        return []
    
    # Count appearances for each eye with position tolerance
    eye_counts = {}
    
    for frame_eyes in eye_history:
        for eye in frame_eyes:
            # Create a position-based key (rounded to reduce exact position dependency)
            ex, ey, ew, eh = eye
            # Round position to create tolerance for slight movements
            pos_key = (round(ex/5)*5, round(ey/5)*5, round(ew/5)*5, round(eh/5)*5)
            eye_counts[pos_key] = eye_counts.get(pos_key, 0) + 1
    
    # Find eyes with highest counts (top 2)
    if eye_counts:
        sorted_eyes = sorted(eye_counts.items(), key=lambda x: x[1], reverse=True)
        consistent_eye_list = []
        
        for i, (pos_key, count) in enumerate(sorted_eyes[:2]):  # Top 2 eyes
            if count >= min_eye_consistency_frames:
                # Return the most recent instance of this eye
                for frame_eyes in reversed(list(eye_history)):
                    for eye in frame_eyes:
                        ex, ey, ew, eh = eye
                        current_pos_key = (round(ex/5)*5, round(ey/5)*5, round(ew/5)*5, round(eh/5)*5)
                        if current_pos_key == pos_key:
                            consistent_eye_list.append(eye)
                            break
                    if len(consistent_eye_list) > i:
                        break
        
        return consistent_eye_list
    return []

def update_heatmap(haar_face_position, current_objects, frame_shape):
    """Update heatmap of squares appearing inside Haar cascade face with padding and center-based retention (OPTIMIZED)"""
    global heatmap, heatmap_haar_position, heatmap_reset_distance
    
    if haar_face_position is None:
        return
    
    haar_x, haar_y, haar_w, haar_h = haar_face_position
    current_haar_center_x = haar_x + haar_w // 2
    current_haar_center_y = haar_y + haar_h // 2
    
    # Check if we need to reset heatmap due to Haar cascade movement
    if heatmap_haar_position is not None:
        old_haar_x, old_haar_y, old_haar_w, old_haar_h = heatmap_haar_position
        old_haar_center_x = old_haar_x + old_haar_w // 2
        old_haar_center_y = old_haar_y + old_haar_h // 2
        
        distance = np.sqrt((current_haar_center_x - old_haar_center_x)**2 + 
                          (current_haar_center_y - old_haar_center_y)**2)
        
        if distance > heatmap_reset_distance:
            # Reset heatmap if Haar cascade moved too far
            heatmap = None
            print(f"Heatmap reset - Haar cascade moved {int(distance)}px")
    
    # Initialize heatmap if needed
    if heatmap is None:
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        heatmap_haar_position = haar_face_position
    
    # Add minimal padding to detection area (5 pixels around Haar cascade)
    padding = 5  # Fixed 5-pixel padding for speed
    padded_x = max(0, haar_x - padding)
    padded_y = max(0, haar_y - padding)
    padded_w = min(frame_shape[1] - padded_x, haar_w + 2 * padding)
    padded_h = min(frame_shape[0] - padded_y, haar_h + 2 * padding)
    
    # FAST HEATMAP CLEANUP: Quickly remove heatmap outside Haar cascade area
    # Create a mask for the Haar cascade area (including padding)
    haar_mask = np.zeros(frame_shape[:2], dtype=np.bool_)
    haar_mask[padded_y:padded_y + padded_h, padded_x:padded_x + padded_w] = True
    
    # Apply aggressive decay to areas outside Haar cascade (90% decay per frame)
    heatmap[~haar_mask] *= 0.1  # 90% decay for areas outside Haar cascade
    
    # Apply normal decay to areas inside Haar cascade
    heatmap[haar_mask] *= 0.98  # 2% decay for areas inside Haar cascade
    
    # Add heat for current objects inside padded area (simplified)
    for obj in current_objects:
        obj_x, obj_y, obj_w, obj_h = obj
        obj_center_x = obj_x + obj_w // 2
        obj_center_y = obj_y + obj_h // 2
        
        # Check if object center is inside padded area
        if (padded_x <= obj_center_x <= padded_x + padded_w and 
            padded_y <= obj_center_y <= padded_y + padded_h):
            
            # Calculate distance from object center to Haar cascade center
            distance_to_center = np.sqrt((obj_center_x - current_haar_center_x)**2 + 
                                       (obj_center_y - current_haar_center_y)**2)
            
            # Simplified heat intensity based on distance
            max_distance = np.sqrt((haar_w//2 + padding)**2 + (haar_h//2 + padding)**2)
            distance_ratio = min(distance_to_center / max_distance, 1.0)
            heat_intensity = 0.4 * (1.0 - distance_ratio * 0.1)  # Higher intensity for longer heatmap life
            
            # Add heat in the object area
            obj_area = heatmap[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w]
            heatmap[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w] = obj_area + heat_intensity
    
    # Update stored Haar cascade position
    heatmap_haar_position = haar_face_position

def calculate_adaptive_haar_interval(haar_face_position):
    """Calculate adaptive Haar detection interval based on position changes"""
    global haar_last_position, haar_position_change_threshold, haar_detection_interval
    
    if haar_face_position is None:
        return 0.5  # Default interval if no face detected
    
    current_x, current_y, current_w, current_h = haar_face_position
    current_center_x = current_x + current_w // 2
    current_center_y = current_y + current_h // 2
    
    if haar_last_position is not None:
        last_x, last_y, last_w, last_h = haar_last_position
        last_center_x = last_x + last_w // 2
        last_center_y = last_y + last_h // 2
        
        # Calculate position change
        position_change = np.sqrt((current_center_x - last_center_x)**2 + (current_center_y - last_center_y)**2)
        
        # Calculate distance from average position
        if average_haar_x is not None and average_haar_y is not None:
            avg_center_x = average_haar_x + (average_haar_w or 0) // 2
            avg_center_y = average_haar_y + (average_haar_h or 0) // 2
            distance_from_avg = np.sqrt((current_center_x - avg_center_x)**2 + (current_center_y - avg_center_y)**2)
        else:
            distance_from_avg = 0
        
        # Adaptive interval based on position change and distance from average
        if position_change > haar_position_change_threshold or distance_from_avg > 30:
            # High movement or far from average - detect more frequently (10 FPS = 0.1s interval)
            adaptive_interval = 0.1
        elif position_change > haar_position_change_threshold * 0.5 or distance_from_avg > 15:
            # Medium movement - detect at 5 FPS (0.2s interval)
            adaptive_interval = 0.2
        elif position_change > haar_position_change_threshold * 0.2 or distance_from_avg > 8:
            # Low movement - detect at 2.5 FPS (0.4s interval)
            adaptive_interval = 0.4
        else:
            # Stable position - detect at 1 FPS (1.0s interval)
            adaptive_interval = 1.0
        
        return adaptive_interval
    else:
        # First detection - use default interval
        return 0.5

def update_haar_averages(haar_face_position):
    """Update average Haar cascade position and size to prevent sudden jumps"""
    global average_haar_x, average_haar_y, average_haar_w, average_haar_h, haar_last_position
    
    if haar_face_position is None:
        return None
    
    haar_x, haar_y, haar_w, haar_h = haar_face_position
    
    # Update averages using exponential moving average
    if average_haar_x is not None:
        average_haar_x = haar_avg_alpha * haar_x + (1 - haar_avg_alpha) * average_haar_x
        average_haar_y = haar_avg_alpha * haar_y + (1 - haar_avg_alpha) * average_haar_y
        average_haar_w = haar_avg_alpha * haar_w + (1 - haar_avg_alpha) * average_haar_w
        average_haar_h = haar_avg_alpha * haar_h + (1 - haar_avg_alpha) * average_haar_h
    else:
        # Initialize averages
        average_haar_x = haar_x
        average_haar_y = haar_y
        average_haar_w = haar_w
        average_haar_h = haar_h
    
    # Update last position for change detection
    haar_last_position = haar_face_position
    
    # Return smoothed Haar cascade position
    return [int(average_haar_x), int(average_haar_y), int(average_haar_w), int(average_haar_h)]

def update_eye_positions(consistent_eyes, current_eyes, face_rect):
    """Update consistent eye positions with maximum speed tracking"""
    if not consistent_eyes or not current_eyes:
        return consistent_eyes
    
    updated_eyes = []
    x, y, w, h = face_rect
    
    for consistent_eye in consistent_eyes:
        # Find closest current eye to this consistent eye
        best_match = None
        best_distance = float('inf')
        
        for current_eye in current_eyes:
            # Calculate distance between eye centers
            cx, cy, cw, ch = current_eye
            consistent_center_x = consistent_eye[0] + consistent_eye[2] // 2
            consistent_center_y = consistent_eye[1] + consistent_eye[3] // 2
            current_center_x = cx + cw // 2
            current_center_y = cy + ch // 2
            
            distance = np.sqrt((consistent_center_x - current_center_x)**2 + 
                             (consistent_center_y - current_center_y)**2)
            
            if distance < best_distance:
                best_distance = distance
                best_match = current_eye
        
        if best_match:
            # Maximum speed update (move directly toward target)
            cx, cy, cw, ch = best_match
            # Use 100% new position and size (maximum speed)
            updated_eyes.append([cx, cy, cw, ch])
        else:
            # Keep the consistent eye if no match found
            updated_eyes.append(consistent_eye)
    
    return updated_eyes

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing (smaller for speed) - OPTIMIZED
    frame = cv2.resize(frame, (400, 300))  # Reduced from 480x360 to 400x300 for speed

    # Adaptive Haar cascade detection based on position changes
    current_time = time.time()
    adaptive_interval = calculate_adaptive_haar_interval(haar_face_position)
    if current_time - last_haar_detection_time >= adaptive_interval:
        haar_face = detect_face_haar(frame)
        if haar_face is not None:
            # Update averages and get smoothed position
            smoothed_haar = update_haar_averages(haar_face)
            haar_face_position = smoothed_haar
            print(f"Haar cascade detected face at {haar_face_position} (adaptive interval: {adaptive_interval:.2f}s)")
        last_haar_detection_time = current_time

    # Use Haar cascade as main detection, then find closest contour with similar size (DYNAMIC AREA)
    current_objects = []
    if haar_face_position is not None:
        # Get Haar cascade face position
        haar_x, haar_y, haar_w, haar_h = haar_face_position
        haar_size = haar_w * haar_h
        
        # Define dynamic search area: Haar cascade + 5 pixel padding
        search_padding = 5  # Minimal padding for speed
        search_x = max(0, haar_x - search_padding)
        search_y = max(0, haar_y - search_padding)
        search_w = min(frame.shape[1] - search_x, haar_w + 2 * search_padding)
        search_h = min(frame.shape[0] - search_y, haar_h + 2 * search_padding)
        
        # Extract search region (much smaller area for speed)
        search_region = frame[search_y:search_y + search_h, search_x:search_x + search_w]
        
        if search_region.size > 0:
            # Convert to HSV for skin tone detection
            hsv_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
            
            # Optimized skin tone range for face detection
            lower_skin = np.array([0, 30, 60], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a mask for skin tone
            mask = cv2.inRange(hsv_region, lower_skin, upper_skin)
            
            # Fast morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            potential_objects = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                contour_size = w * h
                
                # Filter by reasonable size (similar to Haar cascade face)
                size_ratio = contour_size / haar_size
                if 0.3 < size_ratio < 3.0:  # Allow contours 30% to 300% of Haar size
                    # Filter for face-like aspect ratios
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 2.0:
                        # Convert coordinates back to full frame
                        full_x = search_x + x
                        full_y = search_y + y
                        potential_objects.append([full_x, full_y, w, h, contour_size])
            
            # Find the closest contour with similar size to Haar cascade face
            if potential_objects:
                haar_center_x = haar_x + haar_w // 2
                haar_center_y = haar_y + haar_h // 2
                
                best_object = None
                best_score = float('inf')
                
                for obj in potential_objects:
                    obj_x, obj_y, obj_w, obj_h, obj_size = obj
                    obj_center_x = obj_x + obj_w // 2
                    obj_center_y = obj_y + obj_h // 2
                    
                    # Calculate distance
                    distance = np.sqrt((haar_center_x - obj_center_x)**2 + (haar_center_y - obj_center_y)**2)
                    
                    # Calculate size similarity (closer to 1.0 is better)
                    size_similarity = abs(1.0 - (obj_size / haar_size))
                    
                    # Combined score: distance + size similarity (weighted)
                    score = distance + (size_similarity * 50)  # Weight size similarity
                    
                    if score < best_score:
                        best_score = score
                        best_object = [obj_x, obj_y, obj_w, obj_h]
                
                if best_object:
                    current_objects = [best_object]
    else:
        # If no Haar cascade face, use a simple fallback (much faster)
        # Just use a default face area in the center of the frame
        center_x = frame.shape[1] // 2 - 100
        center_y = frame.shape[0] // 2 - 100
        default_w = 200
        default_h = 200
        
        # Ensure the default face area is within frame bounds
        center_x = max(0, min(center_x, frame.shape[1] - default_w))
        center_y = max(0, min(center_y, frame.shape[0] - default_h))
        
        current_objects = [[center_x, center_y, default_w, default_h]]

    # Check if consistency should be reset
    if should_reset_consistency():
        object_history.clear()
        last_consistent_face = None
        print("Consistency reset - no face detected for too long")

    # Update heatmap with current objects
    update_heatmap(haar_face_position, current_objects, frame.shape)
    
    # Add current objects to history
    object_history.append(current_objects)
    
    # Find the most consistent object
    consistent_face = find_most_consistent_object()
    
    # Update consistent face position and size based on most consistent detected object
    if consistent_face:
        consistent_face = update_consistent_face_position(consistent_face, current_objects, consistent_face)
        last_consistent_face = consistent_face
    # If no consistent face but we have a last known face and current objects, try to reconnect
    elif last_consistent_face and len(current_objects) > 0:
        # Find the nearest object to the last known face
        nearest_object = None
        nearest_distance = float('inf')
        
        for obj in current_objects:
            distance = calculate_distance(last_consistent_face, obj)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_object = obj
        
        # If nearest object is close enough, try to re-establish consistency
        if nearest_object and nearest_distance < 150:  # Reconnect threshold
            # Temporarily use the nearest object as the consistent face
            consistent_face = nearest_object
            print(f"Reconnecting lost face to nearest object at distance {int(nearest_distance)}px")
    
    # Draw Haar cascade face position (if available)
    if haar_face_position is not None:
        haar_x, haar_y, haar_w, haar_h = haar_face_position
        # Draw Haar cascade face in blue
        cv2.rectangle(frame, (haar_x, haar_y), (haar_x + haar_w, haar_y + haar_h), (255, 0, 0), 2)
        cv2.putText(frame, 'Haar Face', (haar_x, haar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw search area around Haar cascade face
        search_margin = 50
        search_x = max(0, haar_x - search_margin)
        search_y = max(0, haar_y - search_margin)
        search_w = min(frame.shape[1] - search_x, haar_w + 2 * search_margin)
        search_h = min(frame.shape[0] - search_y, haar_h + 2 * search_margin)
        cv2.rectangle(frame, (search_x, search_y), (search_x + search_w, search_y + search_h), (0, 255, 255), 1)
        cv2.putText(frame, 'Search Area', (search_x, search_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw all detected objects (in gray)
    for obj in current_objects:
        x, y, w, h = obj
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)
        cv2.putText(frame, 'Best Contour', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Draw distance line to Haar cascade face if available
        if haar_face_position is not None:
            haar_x, haar_y, haar_w, haar_h = haar_face_position
            haar_center_x = haar_x + haar_w // 2
            haar_center_y = haar_y + haar_h // 2
            obj_center_x = x + w // 2
            obj_center_y = y + h // 2
            
            distance = np.sqrt((haar_center_x - obj_center_x)**2 + (haar_center_y - obj_center_y)**2)
            
            # Color code the line based on distance (distance-based speed with coordinated movement)
            if distance < 10:
                line_color = (0, 255, 0)  # Green for very close
                speed_text = "MAX SPEED"
            elif distance < 25:
                line_color = (0, 255, 128)  # Light green for close
                speed_text = "HIGH SPEED"
            elif distance < 50:
                line_color = (0, 255, 255)  # Yellow for medium close
                speed_text = "MED SPEED"
            elif distance < 80:
                line_color = (0, 165, 255)  # Orange for medium
                speed_text = "SLOW SPEED"
            elif distance < 120:
                line_color = (0, 100, 255)  # Dark orange for far
                speed_text = "VERY SLOW"
            else:
                line_color = (0, 0, 255)  # Red for very far
                speed_text = "MINIMAL"
            
            # Check for coordinated movement and override color/speed regardless of distance
            if average_haar_x is not None and average_haar_y is not None:
                haar_avg_center_x = average_haar_x + (average_haar_w or 0) // 2
                haar_avg_center_y = average_haar_y + (average_haar_h or 0) // 2
                haar_to_target = np.sqrt((haar_avg_center_x - obj_center_x)**2 + (haar_avg_center_y - obj_center_y)**2)
                
                if abs(haar_to_target - distance) < 30:  # Coordinated movement detected
                    # Calculate movement angle between heatmap and Haar cascade movement vectors
                    
                    movement_angle = 0
                    if (previous_haar_center_x is not None and previous_haar_center_y is not None and 
                        previous_face_center_x is not None and previous_face_center_y is not None):
                        
                        # Haar cascade movement vector
                        haar_movement_dx = haar_avg_center_x - previous_haar_center_x
                        haar_movement_dy = haar_avg_center_y - previous_haar_center_y
                        
                        # Face movement vector
                        face_movement_dx = (x + w // 2) - previous_face_center_x
                        face_movement_dy = (y + h // 2) - previous_face_center_y
                        
                        if (haar_movement_dx != 0 or haar_movement_dy != 0) and (face_movement_dx != 0 or face_movement_dy != 0):
                            haar_magnitude = np.sqrt(haar_movement_dx*haar_movement_dx + haar_movement_dy*haar_movement_dy)
                            face_magnitude = np.sqrt(face_movement_dx*face_movement_dx + face_movement_dy*face_movement_dy)
                            
                            if haar_magnitude > 0 and face_magnitude > 0:
                                dot_product = (haar_movement_dx * face_movement_dx + haar_movement_dy * face_movement_dy) / (haar_magnitude * face_magnitude)
                                dot_product = max(-1.0, min(1.0, dot_product))
                                movement_angle = np.arccos(dot_product) * 180 / np.pi
                    
                    # Ultra-fast angle-based color coding
                    if movement_angle < 5:  # Perfect alignment
                        line_color = (255, 255, 255)  # White for ultra maximum speed
                        speed_text = f"ULTRA MAX ({movement_angle:.1f}°)"
                    elif movement_angle < 10:  # Very straight movement
                        line_color = (0, 255, 255)  # Cyan for extremely fast
                        speed_text = f"EXTREME FAST ({movement_angle:.1f}°)"
                    elif movement_angle < 15:  # Straight movement
                        line_color = (0, 255, 0)  # Green for very fast
                        speed_text = f"VERY FAST ({movement_angle:.1f}°)"
                    elif movement_angle < 25:  # Moderately straight
                        line_color = (0, 255, 128)  # Light green for fast
                        speed_text = f"FAST ({movement_angle:.1f}°)"
                    elif movement_angle < 40:  # Somewhat aligned
                        line_color = (0, 165, 255)  # Orange for medium fast
                        speed_text = f"MED FAST ({movement_angle:.1f}°)"
                    else:  # Less aligned
                        line_color = (0, 255, 0)  # Green for normal coordinated
                        speed_text = f"COORDINATED ({movement_angle:.1f}°)"
            
            cv2.line(frame, (haar_center_x, haar_center_y), (obj_center_x, obj_center_y), line_color, 2)
            cv2.putText(frame, f'{int(distance)}px {speed_text}', 
                       ((haar_center_x + obj_center_x)//2, (haar_center_y + obj_center_y)//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
    
    # Draw the consistent face (in green with "Face" label)
    if consistent_face:
        x, y, w, h = consistent_face
        # Check if this is a reconnected face (not from consistency history)
        is_reconnected = (last_consistent_face and 
                         calculate_distance(consistent_face, last_consistent_face) < 150 and
                         len(object_history) < min_consistency_frames)
        
        if is_reconnected:
            # Draw reconnected face in orange
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 3)  # Orange
            cv2.putText(frame, 'RECONNECTED', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # Draw normal consistent face in green
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, 'FACE', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face center point
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
        
        # Add padding to face area for better eye detection
        padded_face = add_face_padding(consistent_face, frame.shape, padding=20)
        px, py, pw, ph = padded_face
        
        # Extract padded face ROI for eye detection
        padded_face_roi = frame[py:py+ph, px:px+pw]
        if padded_face_roi.size > 0:  # Ensure ROI is valid
            # Start eye FPS tracking
            eye_fps_counter += 1
            
            # Detect eyes in the padded face region using contrast method
            current_eyes, enhanced_face_roi, lighting_mode, mean_brightness = detect_eyes_in_face_contrast(padded_face_roi)
            
            # Add current eyes to history
            eye_history.append(current_eyes)
            
            # Find the most consistent eyes
            consistent_eyes = find_most_consistent_eyes()
            
            # Update consistent eye positions with maximum speed
            if consistent_eyes:
                consistent_eyes = update_eye_positions(consistent_eyes, current_eyes, padded_face)
                last_consistent_eyes = consistent_eyes
            
            # Draw consistent eyes with maximum speed tracking
            if consistent_eyes:
                for i, (ex, ey, ew, eh) in enumerate(consistent_eyes):
                    # Convert eye coordinates from face ROI to full frame coordinates
                    eye_x = px + ex
                    eye_y = py + ey
                    
                    # Draw consistent eye rectangle (thick red for consistent eyes)
                    cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (0, 0, 255), 3)
                    cv2.putText(frame, f'Consistent Eye {i+1}', (eye_x, eye_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw eye center point
                    eye_center_x = eye_x + ew // 2
                    eye_center_y = eye_y + eh // 2
                    cv2.circle(frame, (eye_center_x, eye_center_y), 3, (255, 255, 0), -1)
                    
                    # Draw tracking indicator
                    cv2.putText(frame, 'MAX SPEED', (eye_x, eye_y + eh + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw all detected eyes (thin blue for comparison)
            for (ex, ey, ew, eh) in current_eyes:
                eye_x = px + ex
                eye_y = py + ey
                cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (255, 0, 0), 1)
            
            # Show enhanced face (adjust coordinates for padding)
            draw_eyes_with_enhancement(frame, padded_face, [], enhanced_face_roi, show_enhanced_face)
            
            # Display consistent eye count
            cv2.putText(frame, f'Consistent Eyes: {len(consistent_eyes)}', (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw padding indicator (dashed rectangle around original face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow dashed for original face
            cv2.putText(frame, 'Original Face', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw padded area indicator
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 0), 1)  # Cyan for padded area
            cv2.putText(frame, 'Padded Area', (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw tracking line to most consistent detected object
        if current_objects:
            best_match = None
            best_distance = float('inf')
            for obj in current_objects:
                # Find the object closest to the most consistent object
                distance = calculate_distance(consistent_face, obj)
                if distance < best_distance:
                    best_distance = distance
                    best_match = obj
            
            # Show tracking line to the target object
            if best_match:
                nx, ny, nw, nh = best_match
                target_center_x = nx + nw // 2
                target_center_y = ny + nh // 2
                cv2.line(frame, (center_x, center_y), (target_center_x, target_center_y), (0, 255, 255), 2)
                # Show distance on the line
                cv2.putText(frame, f'{int(best_distance)}px', 
                           ((center_x + target_center_x)//2, (center_y + target_center_y)//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Show last known face position if no current consistent face
    elif last_consistent_face and len(current_objects) > 0:
        x, y, w, h = last_consistent_face
        
        # Find nearest detected object to the last known face
        nearest_object = None
        nearest_distance = float('inf')
        
        for obj in current_objects:
            distance = calculate_distance(last_consistent_face, obj)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_object = obj
        
        # Draw connection line to nearest object
        if nearest_object and nearest_distance < 200:  # Only connect if reasonably close
            nx, ny, nw, nh = nearest_object
            last_center_x = x + w // 2
            last_center_y = y + h // 2
            nearest_center_x = nx + nw // 2
            nearest_center_y = ny + nh // 2
            
            # Draw thick red line connecting lost face to nearest object
            cv2.line(frame, (last_center_x, last_center_y), (nearest_center_x, nearest_center_y), (0, 0, 255), 3)
            
            # Draw arrowhead pointing to nearest object
            cv2.arrowedLine(frame, (last_center_x, last_center_y), (nearest_center_x, nearest_center_y), (0, 0, 255), 3, tipLength=0.3)
            
            # Show distance on the line
            cv2.putText(frame, f'LOST: {int(nearest_distance)}px', 
                       ((last_center_x + nearest_center_x)//2, (last_center_y + nearest_center_y)//2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw last known face position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for last known
        cv2.putText(frame, 'Lost Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Calculate and display eye FPS
    if time.time() - eye_fps_start_time >= 1.0:
        eye_fps = eye_fps_counter
        eye_fps_counter = 0
        eye_fps_start_time = time.time()
    
    # Display detection info
    cv2.putText(frame, f'Detected: {len(current_objects)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Consistent: {"Yes" if consistent_face else "No"}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Reset Counter: {consistency_reset_counter}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add eye detection info if face is detected
    if consistent_face:
        # Extract padded face ROI to get eye count
        padded_face = add_face_padding(consistent_face, frame.shape, padding=20)
        px, py, pw, ph = padded_face
        padded_face_roi = frame[py:py+ph, px:px+pw]
        if padded_face_roi.size > 0:
            current_eyes, _, lighting_mode, mean_brightness = detect_eyes_in_face_contrast(padded_face_roi)
            consistent_eyes = find_most_consistent_eyes()
            cv2.putText(frame, f'Current Eyes: {len(current_eyes)}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0) if len(current_eyes) > 0 else (0, 0, 255), 2)
            cv2.putText(frame, f'Consistent Eyes: {len(consistent_eyes)}', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if len(consistent_eyes) > 0 else (0, 0, 255), 2)
            cv2.putText(frame, f'Eye FPS: {eye_fps}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'MAX SPEED TRACKING', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
             
            # Display adaptive lighting information
            lighting_color = (0, 255, 255) if lighting_mode == "BRIGHT DAYLIGHT" else (0, 255, 0) if lighting_mode == "NORMAL LIGHTING" else (0, 0, 255)
            cv2.putText(frame, f'Lighting: {lighting_mode}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
            cv2.putText(frame, f'Brightness: {int(mean_brightness)}', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
    
    # Show enhanced face display status
    status_color = (0, 255, 0) if show_enhanced_face else (0, 0, 255)
    cv2.putText(frame, f'Enhanced Display: {"ON" if show_enhanced_face else "OFF"}', (10, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Show face padding info
    cv2.putText(frame, 'Face Padding: 20px', (10, 270), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Show performance info
    cv2.putText(frame, 'RESOLUTION: 400x300 (FAST)', (10, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Show hybrid detection method info
    cv2.putText(frame, 'ULTRA FAST: LONG HEATMAP + DYNAMIC AREA', (10, 330), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Show adaptive Haar cascade status
    if haar_face_position is not None:
        adaptive_interval = calculate_adaptive_haar_interval(haar_face_position)
        if adaptive_interval <= 0.1:
            haar_status = f"Haar: ULTRA FAST ({1/adaptive_interval:.0f} FPS)"
            haar_color = (255, 255, 255)  # White for ultra fast
        elif adaptive_interval <= 0.2:
            haar_status = f"Haar: VERY FAST ({1/adaptive_interval:.0f} FPS)"
            haar_color = (0, 255, 255)  # Cyan for very fast
        elif adaptive_interval <= 0.4:
            haar_status = f"Haar: FAST ({1/adaptive_interval:.1f} FPS)"
            haar_color = (0, 255, 0)  # Green for fast
        elif adaptive_interval <= 0.5:
            haar_status = f"Haar: NORMAL ({1/adaptive_interval:.1f} FPS)"
            haar_color = (0, 165, 255)  # Orange for normal
        else:
            haar_status = f"Haar: SLOW ({1/adaptive_interval:.1f} FPS)"
            haar_color = (0, 0, 255)  # Red for slow
    else:
        haar_status = "Haar: SEARCHING (0.5s)"
        haar_color = (0, 0, 255)
    
    cv2.putText(frame, haar_status, (10, 360), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, haar_color, 2)
    
    # Show average Haar cascade position info
    if average_haar_x is not None:
        cv2.putText(frame, f'Avg Haar: ({int(average_haar_x)}, {int(average_haar_y)})', (10, 510), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f'Avg Size: {int(average_haar_w)}x{int(average_haar_h)}', (10, 540), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Show average size information
    if average_face_size is not None:
        cv2.putText(frame, f'Dynamic Avg: {int(average_face_size)}', (10, 390), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        # Calculate and show current size ratio if face is detected
        if consistent_face:
            current_size = consistent_face[2] * consistent_face[3]
            size_ratio = current_size / average_face_size
            ratio_color = (0, 255, 0) if 0.9 <= size_ratio <= 1.3 else (0, 165, 255) if size_ratio < 0.8 else (0, 0, 255)
            cv2.putText(frame, f'Size Ratio: {size_ratio:.2f}', (10, 420), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 2)
    
    # Show distance-based speed control info
    if consistent_face and current_objects:
        # Calculate distance to target
        face_center_x = consistent_face[0] + consistent_face[2] // 2
        face_center_y = consistent_face[1] + consistent_face[3] // 2
        
        best_match = None
        best_distance = float('inf')
        for obj in current_objects:
            obj_center_x = obj[0] + obj[2] // 2
            obj_center_y = obj[1] + obj[3] // 2
            distance = np.sqrt((face_center_x - obj_center_x)**2 + (face_center_y - obj_center_y)**2)
            if distance < best_distance:
                best_distance = distance
                best_match = obj
        
        if best_match:
            # Determine speed level based on distance
            if best_distance < 10:
                speed_level = "MAX SPEED"
                speed_color = (0, 255, 0)
            elif best_distance < 25:
                speed_level = "HIGH SPEED"
                speed_color = (0, 255, 128)
            elif best_distance < 50:
                speed_level = "MED SPEED"
                speed_color = (0, 255, 255)
            elif best_distance < 80:
                speed_level = "SLOW SPEED"
                speed_color = (0, 165, 255)
            elif best_distance < 120:
                speed_level = "VERY SLOW"
                speed_color = (0, 100, 255)
            else:
                speed_level = "MINIMAL"
                speed_color = (0, 0, 255)
            
            cv2.putText(frame, f'Movement: {speed_level}', (10, 570), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)
            cv2.putText(frame, f'Distance: {int(best_distance)}px', (10, 600), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)
            
            # Show coordinated movement status with angle information
            if best_distance > 0:  # Show for all targets when coordinated
                # Check if both heatmap and Haar cascade are moving toward same target
                if average_haar_x is not None and average_haar_y is not None:
                    haar_avg_center_x = average_haar_x + (average_haar_w or 0) // 2
                    haar_avg_center_y = average_haar_y + (average_haar_h or 0) // 2
                    haar_to_target = np.sqrt((haar_avg_center_x - best_match[0] - best_match[2]//2)**2 + 
                                           (haar_avg_center_y - best_match[1] - best_match[3]//2)**2)
                    
                    if abs(haar_to_target - best_distance) < 30:
                        # Calculate movement angle between heatmap and Haar cascade movement vectors
                        
                        movement_angle = 0
                        if (previous_haar_center_x is not None and previous_haar_center_y is not None and 
                            previous_face_center_x is not None and previous_face_center_y is not None):
                            
                            # Haar cascade movement vector
                            haar_movement_dx = haar_avg_center_x - previous_haar_center_x
                            haar_movement_dy = haar_avg_center_y - previous_haar_center_y
                            
                            # Face movement vector
                            face_movement_dx = face_center_x - previous_face_center_x
                            face_movement_dy = face_center_y - previous_face_center_y
                            
                            if (haar_movement_dx != 0 or haar_movement_dy != 0) and (face_movement_dx != 0 or face_movement_dy != 0):
                                haar_magnitude = np.sqrt(haar_movement_dx*haar_movement_dx + haar_movement_dy*haar_movement_dy)
                                face_magnitude = np.sqrt(face_movement_dx*face_movement_dx + face_movement_dy*face_movement_dy)
                                
                                if haar_magnitude > 0 and face_magnitude > 0:
                                    dot_product = (haar_movement_dx * face_movement_dx + haar_movement_dy * face_movement_dy) / (haar_magnitude * face_magnitude)
                                    dot_product = max(-1.0, min(1.0, dot_product))
                                    movement_angle = np.arccos(dot_product) * 180 / np.pi
                        
                        # Ultra-fast angle-based status display (distance doesn't matter when aligned)
                        if movement_angle < 5:
                            cv2.putText(frame, f'ULTRA MAX SPEED <0.1s ({movement_angle:.1f}°)', (10, 630), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif movement_angle < 10:
                            cv2.putText(frame, f'EXTREME FAST SPEED ({movement_angle:.1f}°)', (10, 630), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        elif movement_angle < 15:
                            cv2.putText(frame, f'VERY FAST SPEED ({movement_angle:.1f}°)', (10, 630), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif movement_angle < 25:
                            cv2.putText(frame, f'FAST SPEED ({movement_angle:.1f}°)', (10, 630), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
                        elif movement_angle < 40:
                            cv2.putText(frame, f'MEDIUM FAST ({movement_angle:.1f}°)', (10, 630), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        else:
                            cv2.putText(frame, f'COORDINATED NORMAL ({movement_angle:.1f}°)', (10, 630), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'INDEPENDENT MOVEMENT', (10, 630), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    # Display heatmap overlay
    if heatmap is not None:
        # Normalize heatmap to 0-255 range
        heatmap_normalized = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        
        # Create colored heatmap (red for high heat, blue for low heat)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Blend heatmap with frame (30% opacity)
        alpha = 0.3
        frame_with_heatmap = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Show heatmap status with fast cleanup info
        max_heat = np.max(heatmap)
        cv2.putText(frame_with_heatmap, f'Heatmap: {max_heat:.2f} (FAST CLEANUP)', (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame_with_heatmap, 'Outside Haar: 90% Decay/Frame', (10, 480), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Use the frame with heatmap for display
        frame = frame_with_heatmap

    # Show the video feed
    cv2.imshow('Hybrid Face & Eye Detection', frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        show_enhanced_face = not show_enhanced_face
        print(f"Enhanced face display: {'ON' if show_enhanced_face else 'OFF'}")

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
print("Hybrid Face & Eye Detection App Closed")
