package com.qali.pupil

import android.graphics.Bitmap
import android.graphics.Rect
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
        
        // Face detection parameters
        private const val MIN_FACE_SIZE = 80
        private const val MAX_FACE_SIZE = 400
        private const val GAUSSIAN_BLUR_SIZE = 5
        private const val CANNY_THRESHOLD_1 = 50.0
        private const val CANNY_THRESHOLD_2 = 150.0
        private const val CONTOUR_AREA_THRESHOLD = 3000.0
        private const val ASPECT_RATIO_MIN = 0.6
        private const val ASPECT_RATIO_MAX = 1.4
    }

    private val grayMat = Mat()
    private val blurredMat = Mat()
    private val edgesMat = Mat()
    private val contours = mutableListOf<MatOfPoint>()
    private val hierarchy = Mat()

    fun detectFaces(bitmap: Bitmap): List<RectF> {
        try {
            // Convert bitmap to OpenCV Mat
            val rgbMat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC3)
            Utils.bitmapToMat(bitmap, rgbMat)
            
            // Convert to grayscale
            Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY)
            
            // Apply Gaussian blur to reduce noise
            Imgproc.GaussianBlur(grayMat, blurredMat, Size(GAUSSIAN_BLUR_SIZE.toDouble(), GAUSSIAN_BLUR_SIZE.toDouble()), 0.0)
            
            // Apply Canny edge detection to find edges based on contrast
            Imgproc.Canny(blurredMat, edgesMat, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
            
            // Find contours
            contours.clear()
            Imgproc.findContours(edgesMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            
            // Filter contours to find potential face regions
            val faceRegions = mutableListOf<RectF>()
            
            for (contour in contours) {
                val boundingRect = Imgproc.boundingRect(contour)
                val area = Imgproc.contourArea(contour)
                
                // Filter by area and size constraints
                if (area > CONTOUR_AREA_THRESHOLD && 
                    boundingRect.width >= MIN_FACE_SIZE && 
                    boundingRect.height >= MIN_FACE_SIZE &&
                    boundingRect.width <= MAX_FACE_SIZE && 
                    boundingRect.height <= MAX_FACE_SIZE) {
                    
                    // Check aspect ratio (faces are roughly oval/circular)
                    val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()
                    
                    if (aspectRatio >= ASPECT_RATIO_MIN && aspectRatio <= ASPECT_RATIO_MAX) {
                        // Additional validation: check contour density and shape
                        if (isValidFaceContour(contour, boundingRect)) {
                            faceRegions.add(RectF(
                                boundingRect.x.toFloat(),
                                boundingRect.y.toFloat(),
                                (boundingRect.x + boundingRect.width).toFloat(),
                                (boundingRect.y + boundingRect.height).toFloat()
                            ))
                        }
                    }
                }
            }
            
            // Sort by area (largest first) and return the most likely face
            return faceRegions.sortedByDescending { 
                (it.right - it.left) * (it.bottom - it.top) 
            }.take(1)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in face detection", e)
            return emptyList()
        }
    }
    
    private fun isValidFaceContour(contour: MatOfPoint, boundingRect: Rect): Boolean {
        try {
            // Calculate contour area vs bounding rectangle area ratio
            val contourArea = Imgproc.contourArea(contour)
            val rectArea = (boundingRect.width * boundingRect.height).toDouble()
            val fillRatio = contourArea / rectArea
            
            // Faces typically have a fill ratio between 0.4 and 0.8
            if (fillRatio < 0.4 || fillRatio > 0.8) {
                return false
            }
            
            // Check for convexity - faces tend to have some convex properties
            val hull = MatOfInt()
            Imgproc.convexHull(contour, hull)
            val hullPoints = mutableListOf<Point>()
            val contourArray = contour.toArray()
            
            for (hullIndex in hull.toArray()) {
                hullPoints.add(contourArray[hullIndex])
            }
            
            if (hullPoints.size < 3) return false
            
            val hullArea = abs(Imgproc.contourArea(MatOfPoint(*hullPoints.toTypedArray())))
            val convexityRatio = contourArea / hullArea
            
            // Faces should have a reasonable convexity ratio
            return convexityRatio > 0.7
            
        } catch (e: Exception) {
            Log.e(TAG, "Error validating contour", e)
            return false
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