# Contrast-Based Face Detection Implementation

## Overview

This implementation replaces Google ML Kit's face detection with a custom contrast-based approach using OpenCV. The system detects faces by analyzing edge contrast and contour patterns rather than using pre-trained machine learning models.

## Key Components

### 1. ContrastFaceDetector.kt

The core face detection class that implements contrast-based face detection using OpenCV:

**Key Features:**
- **Edge Detection**: Uses Canny edge detection to identify high-contrast boundaries
- **Contour Analysis**: Finds and analyzes contours formed by contrast differences
- **Face Validation**: Applies geometric and shape-based filters to identify face-like regions
- **Eye Region Estimation**: Calculates eye positions based on face geometry

**Algorithm Steps:**
1. Convert input bitmap to grayscale
2. Apply Gaussian blur to reduce noise
3. Perform Canny edge detection to find contrast edges
4. Find contours from edge map
5. Filter contours by size, aspect ratio, and shape properties
6. Validate face-like characteristics (fill ratio, convexity)
7. Return most likely face region

**Parameters:**
- `MIN_FACE_SIZE`: 80px minimum face dimension
- `MAX_FACE_SIZE`: 400px maximum face dimension
- `CANNY_THRESHOLD_1`: 50.0 (lower threshold for edge detection)
- `CANNY_THRESHOLD_2`: 150.0 (upper threshold for edge detection)
- `CONTOUR_AREA_THRESHOLD`: 3000.0 (minimum contour area)
- `ASPECT_RATIO_MIN/MAX`: 0.6-1.4 (face aspect ratio range)

### 2. OpenCVLoader.kt

Utility class for initializing OpenCV in the Android environment:

**Features:**
- Handles OpenCV library loading and initialization
- Provides callback mechanism for initialization completion
- Supports both internal and external OpenCV libraries

### 3. Updated MainActivity.kt

Modified to use contrast-based detection instead of ML Kit:

**Key Changes:**
- Removed ML Kit face detection dependencies
- Added OpenCV initialization before camera start
- Replaced `FaceAnalyzer` with `ContrastFaceAnalyzer`
- Updated face processing to work with `RectF` instead of ML Kit `Face` objects
- Added proper error handling for OpenCV operations

## How Contrast Detection Works

### Edge Detection Approach

The system uses contrast to detect faces through the following principle:

1. **Facial Features Create Contrast**: Eyes, nose, mouth, and face outline create natural contrast boundaries
2. **Edge Detection**: Canny edge detector identifies these contrast boundaries
3. **Contour Formation**: Connected edge pixels form contours around facial features
4. **Face Recognition**: Contours with face-like properties are identified as potential faces

### Validation Criteria

A detected contour is considered a face if it meets these criteria:

1. **Size Constraints**: Between 80-400 pixels in width/height
2. **Aspect Ratio**: Between 0.6-1.4 (roughly oval shape)
3. **Fill Ratio**: 0.4-0.8 (contour area vs bounding rectangle)
4. **Convexity Ratio**: >0.7 (face-like convex properties)

### Eye Region Estimation

Once a face is detected, eye regions are estimated using facial geometry:

- Eyes located at ~35% from top of face
- Left eye at ~25% from left edge
- Right eye at ~75% from left edge
- Eye size estimated as ~15% of face width

## Advantages of Contrast-Based Detection

1. **No ML Model Dependency**: Eliminates need for pre-trained models
2. **Lightweight**: Lower computational overhead than neural networks
3. **Privacy Focused**: No data sent to external services
4. **Customizable**: Easy to tune parameters for different conditions
5. **Real-time Performance**: Fast enough for live camera processing

## Usage

```kotlin
// Initialize the detector
val contrastFaceDetector = ContrastFaceDetector()

// Detect faces from bitmap
val faces = contrastFaceDetector.detectFaces(bitmap)

if (faces.isNotEmpty()) {
    val faceRect = faces[0] // Get largest face
    
    // Get eye regions for gaze tracking
    val eyeRegions = contrastFaceDetector.getEyeRegions(faceRect)
    if (eyeRegions != null) {
        val (leftEye, rightEye) = eyeRegions
        // Process eye regions for gaze estimation
    }
}

// Release resources when done
contrastFaceDetector.release()
```

## Configuration

The detection can be tuned by adjusting parameters in `ContrastFaceDetector`:

- **For Better Sensitivity**: Lower `CANNY_THRESHOLD_1` and `CONTOUR_AREA_THRESHOLD`
- **For More Precision**: Increase thresholds and tighten aspect ratio range
- **For Different Face Sizes**: Adjust `MIN_FACE_SIZE` and `MAX_FACE_SIZE`

## Dependencies

- OpenCV for Android: `com.quickbirdstudios:opencv:4.5.3.0`
- CameraX for camera handling
- TensorFlow Lite for gaze estimation (unchanged)

## Performance Considerations

- Processing time scales with image resolution
- Optimal performance at 640x480 camera resolution
- Consider reducing image size for faster processing if needed
- OpenCV operations are optimized for mobile devices

## Integration Notes

The contrast-based detector integrates seamlessly with the existing gaze tracking pipeline:

1. Camera captures frames via CameraX
2. ContrastFaceDetector identifies face regions
3. Eye regions extracted based on face geometry
4. Eye images fed to existing TensorFlow Lite gaze model
5. Gaze coordinates displayed on overlay

This maintains the same user experience while replacing the face detection mechanism with a contrast-based approach.