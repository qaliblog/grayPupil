package com.qali.pupil

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.*
import kotlin.math.*

class ContrastFaceDetector {
    companion object {
        private const val TAG = "ContrastFaceDetector"
        
        // Face detection parameters - Made more permissive
        private const val MIN_FACE_SIZE = 50        // Reduced from 80
        private const val MAX_FACE_SIZE = 600       // Increased from 400
        private const val GAUSSIAN_BLUR_SIZE = 3    // Reduced from 5 for better edge preservation
        private const val CANNY_THRESHOLD_1 = 30.0  // Reduced from 50 for more sensitive edge detection
        private const val CANNY_THRESHOLD_2 = 100.0 // Reduced from 150
        private const val CONTOUR_AREA_THRESHOLD = 1000.0  // Reduced from 3000
        private const val ASPECT_RATIO_MIN = 0.5    // More permissive from 0.6
        private const val ASPECT_RATIO_MAX = 2.0    // More permissive from 1.4
        private const val FILL_RATIO_MIN = 0.2      // More permissive from 0.4
        private const val FILL_RATIO_MAX = 0.9      // More permissive from 0.8
        private const val CONVEXITY_RATIO_MIN = 0.5 // More permissive from 0.7
    }

    private val grayMat = Mat()
    private val blurredMat = Mat()
    private val edgesMat = Mat()
    private val contours = mutableListOf<MatOfPoint>()
    private val hierarchy = Mat()

    fun detectFaces(bitmap: Bitmap): List<RectF> {
        try {
            Log.d(TAG, "Starting face detection on bitmap: ${bitmap.width}x${bitmap.height}")
            
            // Convert bitmap to OpenCV Mat
            val rgbMat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC3)
            Utils.bitmapToMat(bitmap, rgbMat)
            
            // Convert to grayscale
            Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY)
            Log.d(TAG, "Converted to grayscale: ${grayMat.rows()}x${grayMat.cols()}")
            
            // Apply Gaussian blur to reduce noise
            Imgproc.GaussianBlur(grayMat, blurredMat, Size(GAUSSIAN_BLUR_SIZE.toDouble(), GAUSSIAN_BLUR_SIZE.toDouble()), 0.0)
            
            // Apply Canny edge detection to find edges based on contrast
            Imgproc.Canny(blurredMat, edgesMat, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
            
            // Find contours
            contours.clear()
            Imgproc.findContours(edgesMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            
            Log.d(TAG, "Found ${contours.size} contours")
            
            // Filter contours to find potential face regions
            val faceRegions = mutableListOf<RectF>()
            var validContourCount = 0
            
            for (i in contours.indices) {
                val contour = contours[i]
                val boundingRect = Imgproc.boundingRect(contour)
                val area = Imgproc.contourArea(contour)
                
                Log.d(TAG, "Contour $i: area=$area, rect=${boundingRect.width}x${boundingRect.height}")
                
                // Filter by area and size constraints
                if (area > CONTOUR_AREA_THRESHOLD && 
                    boundingRect.width >= MIN_FACE_SIZE && 
                    boundingRect.height >= MIN_FACE_SIZE &&
                    boundingRect.width <= MAX_FACE_SIZE && 
                    boundingRect.height <= MAX_FACE_SIZE) {
                    
                    validContourCount++
                    Log.d(TAG, "Contour $i passed size filter")
                    
                    // Check aspect ratio (faces are roughly oval/circular)
                    val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()
                    Log.d(TAG, "Contour $i aspect ratio: $aspectRatio")
                    
                    if (aspectRatio >= ASPECT_RATIO_MIN && aspectRatio <= ASPECT_RATIO_MAX) {
                        Log.d(TAG, "Contour $i passed aspect ratio filter")
                        
                        // Additional validation: check contour density and shape
                        if (isValidFaceContour(contour, boundingRect)) {
                            Log.d(TAG, "Contour $i passed face validation - adding as face region")
                            
                            faceRegions.add(RectF(
                                boundingRect.x.toFloat(),
                                boundingRect.y.toFloat(),
                                (boundingRect.x + boundingRect.width).toFloat(),
                                (boundingRect.y + boundingRect.height).toFloat()
                            ))
                        } else {
                            Log.d(TAG, "Contour $i failed face validation")
                        }
                    } else {
                        Log.d(TAG, "Contour $i failed aspect ratio filter")
                    }
                } else {
                    Log.d(TAG, "Contour $i failed size filter: area=$area (min=${CONTOUR_AREA_THRESHOLD}), size=${boundingRect.width}x${boundingRect.height}")
                }
            }
            
            Log.d(TAG, "Valid contours after size filter: $validContourCount")
            Log.d(TAG, "Final face regions found: ${faceRegions.size}")
            
            // Sort by area (largest first) and return top candidates
            val sortedFaces = faceRegions.sortedByDescending { 
                (it.right - it.left) * (it.bottom - it.top) 
            }.take(3) // Take top 3 candidates instead of just 1
            
            Log.d(TAG, "Returning ${sortedFaces.size} face regions")
            return sortedFaces
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in face detection", e)
            return emptyList()
        }
    }
    
    private fun isValidFaceContour(contour: MatOfPoint, boundingRect: org.opencv.core.Rect): Boolean {
        try {
            // Calculate contour area vs bounding rectangle area ratio
            val contourArea = Imgproc.contourArea(contour)
            val rectArea = (boundingRect.width * boundingRect.height).toDouble()
            val fillRatio = contourArea / rectArea
            
            Log.d(TAG, "Fill ratio: $fillRatio (contour area: $contourArea, rect area: $rectArea)")
            
            // More permissive fill ratio for faces
            if (fillRatio < FILL_RATIO_MIN || fillRatio > FILL_RATIO_MAX) {
                Log.d(TAG, "Fill ratio validation failed: $fillRatio not in range [$FILL_RATIO_MIN, $FILL_RATIO_MAX]")
                return false
            }
            
            // Check for convexity - faces tend to have some convex properties
            val hull = MatOfInt()
            Imgproc.convexHull(contour, hull)
            val hullPoints = mutableListOf<Point>()
            val contourArray = contour.toArray()
            
            for (hullIndex in hull.toArray()) {
                if (hullIndex < contourArray.size) {
                    hullPoints.add(contourArray[hullIndex])
                }
            }
            
            if (hullPoints.size < 3) {
                Log.d(TAG, "Convexity validation failed: insufficient hull points")
                return false
            }
            
            val hullArea = abs(Imgproc.contourArea(MatOfPoint(*hullPoints.toTypedArray())))
            val convexityRatio = if (hullArea > 0) contourArea / hullArea else 0.0
            
            Log.d(TAG, "Convexity ratio: $convexityRatio")
            
            // More permissive convexity ratio
            val isValidConvexity = convexityRatio > CONVEXITY_RATIO_MIN
            if (!isValidConvexity) {
                Log.d(TAG, "Convexity validation failed: $convexityRatio <= $CONVEXITY_RATIO_MIN")
            }
            
            return isValidConvexity
            
        } catch (e: Exception) {
            Log.e(TAG, "Error validating contour", e)
            return false
        }
    }
    
    // Alternative simpler detection method for debugging
    fun detectFacesSimple(bitmap: Bitmap): List<RectF> {
        try {
            Log.d(TAG, "Starting SIMPLE face detection")
            
            val rgbMat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC3)
            Utils.bitmapToMat(bitmap, rgbMat)
            
            Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY)
            
            // Apply histogram equalization to improve contrast
            Imgproc.equalizeHist(grayMat, grayMat)
            
            // Very permissive edge detection
            Imgproc.Canny(grayMat, edgesMat, 15.0, 45.0)
            
            contours.clear()
            Imgproc.findContours(edgesMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            
            Log.d(TAG, "Simple method found ${contours.size} contours")
            
            val faceRegions = mutableListOf<RectF>()
            var validContours = 0
            
            for (i in contours.indices) {
                val contour = contours[i]
                val boundingRect = Imgproc.boundingRect(contour)
                val area = Imgproc.contourArea(contour)
                
                Log.d(TAG, "Simple contour $i: area=$area, size=${boundingRect.width}x${boundingRect.height}")
                
                // Very basic filtering - focus on face-like sizes
                if (area > 1000 && 
                    boundingRect.width >= 60 && 
                    boundingRect.height >= 60 &&
                    boundingRect.width <= 500 && 
                    boundingRect.height <= 500) {
                    
                    // Basic aspect ratio check (faces are roughly 0.7 to 1.3 ratio)
                    val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()
                    if (aspectRatio > 0.5 && aspectRatio < 2.0) {
                        validContours++
                        faceRegions.add(RectF(
                            boundingRect.x.toFloat(),
                            boundingRect.y.toFloat(),
                            (boundingRect.x + boundingRect.width).toFloat(),
                            (boundingRect.y + boundingRect.height).toFloat()
                        ))
                        Log.d(TAG, "Simple contour $i accepted as face candidate")
                    } else {
                        Log.d(TAG, "Simple contour $i rejected - bad aspect ratio: $aspectRatio")
                    }
                } else {
                    Log.d(TAG, "Simple contour $i rejected - size/area filter")
                }
            }
            
            Log.d(TAG, "Simple method: $validContours valid contours, returning ${faceRegions.size} regions")
            
            // Sort by area and return largest candidates
            return faceRegions.sortedByDescending { 
                (it.right - it.left) * (it.bottom - it.top) 
            }.take(3)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in simple face detection", e)
            return emptyList()
        }
    }
    
    fun getEyeRegions(faceRect: RectF): Pair<RectF, RectF>? {
        try {
            // Estimate eye positions based on face geometry
            val faceWidth = faceRect.right - faceRect.left
            val faceHeight = faceRect.bottom - faceRect.top
            
            // Eyes are typically located at about 1/3 from the top and 1/4 from each side
            val eyeY = faceRect.top + faceHeight * 0.35f
            val eyeSize = faceWidth * 0.15f
            
            val leftEyeX = faceRect.left + faceWidth * 0.25f
            val rightEyeX = faceRect.left + faceWidth * 0.75f
            
            val leftEyeRect = RectF(
                leftEyeX - eyeSize,
                eyeY - eyeSize,
                leftEyeX + eyeSize,
                eyeY + eyeSize
            )
            
            val rightEyeRect = RectF(
                rightEyeX - eyeSize,
                eyeY - eyeSize,
                rightEyeX + eyeSize,
                eyeY + eyeSize
            )
            
            return Pair(leftEyeRect, rightEyeRect)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error getting eye regions", e)
            return null
        }
    }
    
    fun release() {
        grayMat.release()
        blurredMat.release()
        edgesMat.release()
        hierarchy.release()
    }
}