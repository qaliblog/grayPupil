# Face Detection Troubleshooting Guide

## Problem: No faces detected, no contrast shown

If the contrast-based face detection isn't working, follow these debugging steps:

## 1. Check Logs

Enable debug logging to see what's happening:

```bash
adb logcat | grep "ContrastFaceDetector\|EyeTracking"
```

Look for these key log messages:
- `Starting face detection on bitmap: WxH` - Confirms detection is running
- `Found X contours` - Shows if edge detection is working
- `Contour X: area=Y, rect=WxH` - Shows contour details
- `Simple detection found X faces` - Shows if simple method works

## 2. Visual Debug Information

The updated overlay now shows:
- **Green rectangles**: Detected face regions
- **Yellow text**: "Face 0", "Face 1", etc. for each detected face
- **Yellow counter**: "Faces: X" showing total count
- **Red circle**: Gaze point (if face detection works)

## 3. Common Issues & Solutions

### Issue 1: No contours found
**Symptoms**: Log shows "Found 0 contours"
**Causes**: 
- Low contrast in image
- Camera exposure too bright/dark
- OpenCV not initialized

**Solutions**:
- Test in better lighting conditions
- Check OpenCV initialization logs
- Try moving closer to camera

### Issue 2: Contours found but no valid faces
**Symptoms**: Log shows "Found X contours" but "Simple detection found 0 faces"
**Causes**: 
- Contours too small/large
- Wrong aspect ratios
- Validation too strict

**Solutions**:
- Lower thresholds in `ContrastFaceDetector`
- Check contour sizes in logs
- Try the simple detection method

### Issue 3: Camera/Bitmap conversion issues
**Symptoms**: "Converted ImageProxy to bitmap: nullxnull"
**Causes**:
- Camera format not supported
- Bitmap conversion failing

**Solutions**:
- Check camera permissions
- Verify CameraX setup
- Test with different camera resolutions

## 4. Tuning Parameters

### For Low Light Conditions:
```kotlin
private const val CANNY_THRESHOLD_1 = 20.0  // Lower for more edges
private const val CANNY_THRESHOLD_2 = 80.0   // Lower for more edges
private const val CONTOUR_AREA_THRESHOLD = 500.0  // Smaller contours
```

### For High Contrast Scenes:
```kotlin
private const val CANNY_THRESHOLD_1 = 50.0  // Higher to reduce noise
private const val CANNY_THRESHOLD_2 = 150.0 // Higher to reduce noise
private const val CONTOUR_AREA_THRESHOLD = 2000.0  // Larger contours only
```

### For Different Face Sizes:
```kotlin
private const val MIN_FACE_SIZE = 30        // Smaller faces
private const val MAX_FACE_SIZE = 800       // Larger faces
```

## 5. Testing Steps

### Step 1: Verify Basic Setup
1. Check app permissions (camera)
2. Verify OpenCV loads successfully
3. Confirm camera preview is working

### Step 2: Test Simple Detection
The app now tries simple detection first - this should catch basic shapes:
- Lower thresholds
- Minimal validation
- Should show green rectangles around any large contours

### Step 3: Check Edge Detection
- Point camera at high-contrast objects (text, doors, windows)
- Should see contours in logs even if not faces
- If no contours, lighting/contrast is the issue

### Step 4: Face-Specific Testing
- Test with well-lit face straight-on to camera
- Distance should be 1-3 feet from camera
- Face should be 20-50% of frame size

## 6. Alternative Detection Method

If the main algorithm still fails, you can force use the simple method:

```kotlin
// In MainActivity, replace detectFaces() call:
val faces = contrastFaceDetector.detectFacesSimple(bitmap)
```

This uses:
- Very low Canny thresholds (20, 60)
- Minimal size filtering (30x30 to 800x800 pixels)
- No shape validation
- Should detect any large contrasted object

## 7. Expected Log Output (Working)

```
D/EyeTracking: Processing frame: 640x480
D/ContrastFaceDetector: Starting SIMPLE face detection
D/ContrastFaceDetector: Simple method found 127 contours
D/ContrastFaceDetector: Simple method returning 3 regions
D/EyeTracking: Simple detection found 3 faces
D/EyeTracking: Processing face: RectF(100.0, 150.0, 300.0, 350.0)
```

## 8. Emergency Fallback

If contrast detection completely fails, you can temporarily revert to a basic rectangle in the center of the screen:

```kotlin
// Add this method to ContrastFaceDetector for testing:
fun detectFacesFallback(bitmap: Bitmap): List<RectF> {
    val centerX = bitmap.width / 2f
    val centerY = bitmap.height / 2f
    val size = 200f
    
    return listOf(RectF(
        centerX - size/2, centerY - size/2,
        centerX + size/2, centerY + size/2
    ))
}
```

This will create a fake face region in the center to test the rest of the gaze tracking pipeline.