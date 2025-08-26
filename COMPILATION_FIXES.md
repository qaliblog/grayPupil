# Compilation Fixes for OpenCV Rect Issues

## Issues Encountered

You encountered these compilation errors when building the contrast-based face detection implementation:

```
e: Expression 'width' of type 'Int' cannot be invoked as a function. The function 'invoke()' is not found
e: Expression 'height' of type 'Int' cannot be invoked as a function. The function 'invoke()' is not found
```

## Root Cause

The original error message was misleading. The actual issue was:
- **OpenCV Rect uses `width` and `height` as PROPERTIES (Int values)**
- **Not as methods like Android's Rect class**

## Correct Fixes Applied

### 1. Fixed Import Statement ✅

**Problem**: Import conflict between `android.graphics.Rect` and `org.opencv.core.Rect`

**Solution**: Removed the Android Rect import since we only need OpenCV's Rect
```kotlin
// BEFORE
import android.graphics.Rect
import android.graphics.RectF

// AFTER  
import android.graphics.RectF
```

### 2. Fixed Type Declaration ✅

**Problem**: Function parameter type was ambiguous between Android and OpenCV Rect

**Solution**: Explicitly specified OpenCV Rect type
```kotlin
// BEFORE
private fun isValidFaceContour(contour: MatOfPoint, boundingRect: Rect): Boolean

// AFTER
private fun isValidFaceContour(contour: MatOfPoint, boundingRect: org.opencv.core.Rect): Boolean
```

### 3. CORRECTED: Properties vs Methods ✅

**Problem**: Initially tried to use methods, but OpenCV Rect uses properties

**CORRECT Solution**: Use properties for width and height
```kotlin
// INCORRECT - Method calls (caused "cannot be invoked as function" error)
boundingRect.width() >= MIN_FACE_SIZE
boundingRect.height() >= MIN_FACE_SIZE

// CORRECT - Property access  
boundingRect.width >= MIN_FACE_SIZE
boundingRect.height >= MIN_FACE_SIZE
val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()
val rectArea = (boundingRect.width * boundingRect.height).toDouble()
```

### 4. Fixed All Property Access ✅

**Correct usage throughout the file**:
```kotlin
// Size constraints - PROPERTIES
boundingRect.width >= MIN_FACE_SIZE
boundingRect.height >= MIN_FACE_SIZE

// Aspect ratio calculation - PROPERTIES
val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()

// Area calculation - PROPERTIES
val rectArea = (boundingRect.width * boundingRect.height).toDouble()

// RectF construction - PROPERTIES + PROPERTIES
(boundingRect.x + boundingRect.width).toFloat()
(boundingRect.y + boundingRect.height).toFloat()
```

## Key Differences: OpenCV vs Android Rect

| Property | OpenCV Rect | Android Rect |
|----------|-------------|--------------|
| Width | `width` **property** | `width()` **method** |
| Height | `height` **property** | `height()` **method** |
| X coordinate | `x` property | `left` property |
| Y coordinate | `y` property | `top` property |
| Package | `org.opencv.core` | `android.graphics` |

## Corrected Understanding

**OpenCV Rect API**:
- `width` and `height` are **Int properties**
- `x` and `y` are **Int properties** 
- **No method calls needed**

**Android Rect API**:
- `width()` and `height()` are **methods**
- `left`, `top`, `right`, `bottom` are **properties**

## Summary

The main lessons learned:

1. **API Documentation Matters**: Don't assume API similarity between libraries
2. **OpenCV Rect**: All dimensions are **properties** (`width`, `height`, `x`, `y`)
3. **Android Rect**: Dimensions are **methods** (`width()`, `height()`)
4. **Import Conflicts**: Always use fully qualified names when necessary
5. **Error Messages**: "Function invocation expected" was misleading - the real issue was trying to call properties as functions

**Final Resolution**:
- ✅ Use `boundingRect.width` (property)
- ✅ Use `boundingRect.height` (property)  
- ✅ Use `boundingRect.x` (property)
- ✅ Use `boundingRect.y` (property)
- ❌ Don't use `boundingRect.width()` (not a method)

The contrast-based face detection should now compile successfully with OpenCV for Android.