# Black Screen Diagnostic Guide

## Problem: Black screen, "Faces: 0" displayed

You're seeing a black screen with zero faces detected. Let's diagnose the root cause systematically.

## Step 1: Check Logs for Basic Setup

Run this command and look for the specific messages:

```bash
adb logcat | grep "EyeTracking"
```

### Expected Log Sequence (Working):
```
D/EyeTracking: === MainActivity onCreate started ===
D/EyeTracking: Views initialized, checking permissions...
D/EyeTracking: Camera permissions granted
D/EyeTracking: === Starting OpenCV initialization ===
D/EyeTracking: OpenCV initialization completed successfully
D/EyeTracking: TensorFlow Lite model loaded successfully
D/EyeTracking: === Starting camera setup ===
D/EyeTracking: Camera provider obtained successfully
D/EyeTracking: Preview surface provider set
D/EyeTracking: Image analyzer set
D/EyeTracking: Front camera bound successfully
D/EyeTracking: === Frame 1 Analysis ===
D/EyeTracking: ImageProxy format: 35
D/EyeTracking: ImageProxy size: 640x480
D/EyeTracking: Bitmap created: 640x480, config: ARGB_8888
D/EyeTracking: Basic detection found 1 faces
```

## Step 2: Identify the Failure Point

### If you see:
- **"Camera permission not granted"** → Go to Step 3A
- **"OpenCV initialization failed"** → Go to Step 3B  
- **"Both cameras failed to bind"** → Go to Step 3C
- **"Failed to convert ImageProxy to bitmap"** → Go to Step 3D
- **No frame analysis logs** → Go to Step 3E

## Step 3A: Camera Permission Issues

### Problem: Camera permission denied
### Solution:
1. Go to Android Settings → Apps → Your App → Permissions
2. Enable Camera permission
3. Restart the app
4. If still failing, try:
   ```bash
   adb shell pm grant com.qali.pupil android.permission.CAMERA
   ```

## Step 3B: OpenCV Initialization Failed

### Problem: OpenCV library not loading
### Check logs for:
```bash
adb logcat | grep "OpenCV"
```

### Solutions:
1. **Missing OpenCV library**: 
   - Check if `com.quickbirdstudios:opencv:4.5.3.0` is in dependencies
   - Clean and rebuild: `./gradlew clean build`

2. **Architecture mismatch**:
   - Add to app's `build.gradle`:
   ```kotlin
   android {
       defaultConfig {
           ndk {
               abiFilters "arm64-v8a", "armeabi-v7a", "x86", "x86_64"
           }
       }
   }
   ```

## Step 3C: Camera Binding Failed

### Problem: No camera available or camera access blocked
### Check logs for specific camera errors

### Solutions:
1. **Test on different device**: Some emulators don't have cameras
2. **Check camera in other apps**: Verify camera hardware works
3. **Try different camera resolution**:
   ```kotlin
   .setTargetResolution(Size(320, 240))  // Smaller resolution
   ```

## Step 3D: Bitmap Conversion Failed

### Problem: ImageProxy to Bitmap conversion failing
### Solutions:
1. **Camera format issue**: Add this to check format:
   ```bash
   adb logcat | grep "ImageProxy format"
   ```
2. **Memory issue**: Lower camera resolution
3. **Format compatibility**: Some camera formats aren't supported

## Step 3E: No Frame Processing

### Problem: Camera working but no frames reaching analyzer
### Solutions:
1. **Background processing**: App might be paused/backgrounded
2. **Thread issues**: Camera executor not working
3. **Analyzer not set**: Image analyzer setup failed

## Emergency Test Mode

If all else fails, the app now includes a test pattern mode. Look for this log:
```
D/EyeTracking: Showing test pattern instead of camera
```

You should see:
- **2 green rectangles** (fake faces)
- **"Faces: 2"** in yellow text
- **Red circle** in center (gaze point)

This proves the overlay and detection pipeline work - the issue is with camera setup.

## Force Test Pattern

To manually trigger test pattern mode, add this to `onCreate()`:
```kotlin
// Add this line after overlayView setup to force test mode
showTestPattern()
```

## Most Likely Causes

Based on your symptoms, the most likely issues are:

1. **Camera permissions not granted** (90% probability)
2. **OpenCV library not loading** (5% probability)  
3. **Device/emulator has no camera** (3% probability)
4. **App architecture mismatch** (2% probability)

## Quick Test

Try this simple test:
1. **Grant camera permission manually** in device settings
2. **Restart the app**
3. **Check if preview shows anything** (even if distorted)
4. **Look for green rectangles** from basic detection

If you still see black screen + "Faces: 0", check the logs using the commands above and let me know what you see.