package com.qali.pupil

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import kotlin.math.*

class ContrastFaceDetector(private val context: Context) {
    companion object {
        private const val TAG = "ContrastFaceDetector"
        
        // Haar cascade parameters (standard Python OpenCV values)
        private const val SCALE_FACTOR = 1.1
        private const val MIN_NEIGHBORS = 3
        private const val MIN_SIZE_RATIO = 0.1 // Minimum face size as ratio of image
        
        // Contrast enhancement parameters
        private const val CONTRAST_ALPHA = 1.5  // Contrast multiplier
        private const val BRIGHTNESS_BETA = 10  // Brightness offset
        private const val HISTOGRAM_CLIP_LIMIT = 2.0
        private const val CLAHE_GRID_SIZE = 8
    }

    private val grayMat = Mat()
    private val enhancedMat = Mat()
    private val heatmapMat = Mat()
    private val faces = MatOfRect()
    private var cascadeClassifier: CascadeClassifier? = null

    init {
        loadHaarCascade()
    }

    private fun loadHaarCascade() {
        try {
            // Load the Haar cascade from assets (standard frontal face cascade)
            val inputStream: InputStream = context.assets.open("haarcascade_frontalface_alt.xml")
            val cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE)
            val cascadeFile = File(cascadeDir, "haarcascade_frontalface_alt.xml")
            
            val outputStream = FileOutputStream(cascadeFile)
            val buffer = ByteArray(4096)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
            }
            inputStream.close()
            outputStream.close()
            
            cascadeClassifier = CascadeClassifier(cascadeFile.absolutePath)
            
            if (cascadeClassifier?.empty() == true) {
                Log.e(TAG, "Failed to load Haar cascade classifier")
                cascadeClassifier = null
            } else {
                Log.d(TAG, "Haar cascade classifier loaded successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading Haar cascade", e)
            cascadeClassifier = null
        }
    }

    fun detectFaces(bitmap: Bitmap): List<RectF> {
        try {
            Log.d(TAG, "Starting face detection on ${bitmap.width}x${bitmap.height} image")
            
            // Convert bitmap to OpenCV Mat
            val rgbMat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC3)
            Utils.bitmapToMat(bitmap, rgbMat)
            
            // Convert to grayscale (standard OpenCV flow)
            Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY)
            
            // Apply contrast enhancement and create heatmap (Python pattern)
            val faceRegions = mutableListOf<RectF>()
            
            // Method 1: Try Haar cascade first (if available)
            if (cascadeClassifier != null) {
                val haarFaces = detectWithHaarCascade()
                if (haarFaces.isNotEmpty()) {
                    Log.d(TAG, "Haar cascade found ${haarFaces.size} faces")
                    faceRegions.addAll(haarFaces)
                }
            }
            
            // Method 2: Enhance with contrast-based detection
            val contrastFaces = detectWithContrastHeatmap()
            if (contrastFaces.isNotEmpty()) {
                Log.d(TAG, "Contrast heatmap found ${contrastFaces.size} additional regions")
                faceRegions.addAll(contrastFaces)
            }
            
            // Method 3: Fallback to edge-based detection
            if (faceRegions.isEmpty()) {
                val edgeFaces = detectWithEdges()
                if (edgeFaces.isNotEmpty()) {
                    Log.d(TAG, "Edge detection found ${edgeFaces.size} regions")
                    faceRegions.addAll(edgeFaces)
                }
            }
            
            // Remove overlapping detections and return best candidates
            val filteredFaces = filterOverlappingFaces(faceRegions)
            Log.d(TAG, "Final result: ${filteredFaces.size} faces after filtering")
            
            return filteredFaces.take(2) // Limit to 2 best detections
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in face detection", e)
            return emptyList()
        }
    }
    
    private fun detectWithHaarCascade(): List<RectF> {
        val faceRegions = mutableListOf<RectF>()
        
        try {
            val classifier = cascadeClassifier ?: return faceRegions
            
            // Calculate minimum face size (standard Python approach)
            val minSize = Size(
                (grayMat.cols() * MIN_SIZE_RATIO).toDouble(),
                (grayMat.rows() * MIN_SIZE_RATIO).toDouble()
            )
            
            // Detect faces using standard Haar cascade parameters
            classifier.detectMultiScale(
                grayMat,
                faces,
                SCALE_FACTOR,
                MIN_NEIGHBORS,
                0,
                minSize,
                Size()
            )
            
            val facesArray = faces.toArray()
            Log.d(TAG, "Haar cascade detected ${facesArray.size} faces")
            
            for (face in facesArray) {
                faceRegions.add(RectF(
                    face.x.toFloat(),
                    face.y.toFloat(),
                    (face.x + face.width).toFloat(),
                    (face.y + face.height).toFloat()
                ))
                Log.d(TAG, "Haar face: ${face.x},${face.y},${face.width}x${face.height}")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in Haar cascade detection", e)
        }
        
        return faceRegions
    }
    
    private fun detectWithContrastHeatmap(): List<RectF> {
        val faceRegions = mutableListOf<RectF>()
        
        try {
            // Step 1: Apply contrast enhancement (Python cv2.convertScaleAbs equivalent)
            grayMat.convertTo(enhancedMat, CvType.CV_8UC1, CONTRAST_ALPHA, BRIGHTNESS_BETA.toDouble())
            
            // Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            val clahe = Imgproc.createCLAHE(HISTOGRAM_CLIP_LIMIT, Size(CLAHE_GRID_SIZE.toDouble(), CLAHE_GRID_SIZE.toDouble()))
            clahe.apply(enhancedMat, heatmapMat)
            
            // Step 3: Create thermal/heatmap visualization
            val thermalMat = Mat()
            Imgproc.applyColorMap(heatmapMat, thermalMat, Imgproc.COLORMAP_JET)
            
            // Step 4: Find regions with high thermal variance (face-like regions)
            val kernelSize = 31
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kernelSize.toDouble(), kernelSize.toDouble()))
            
            val morphMat = Mat()
            Imgproc.morphologyEx(heatmapMat, morphMat, Imgproc.MORPH_TOPHAT, kernel)
            
            // Step 5: Threshold to find significant regions
            val threshMat = Mat()
            Imgproc.threshold(morphMat, threshMat, 30.0, 255.0, Imgproc.THRESH_BINARY)
            
            // Step 6: Find contours in thermal regions
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(threshMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            
            Log.d(TAG, "Heatmap method found ${contours.size} thermal regions")
            
            // Step 7: Filter contours for face-like characteristics
            for (contour in contours) {
                val boundingRect = Imgproc.boundingRect(contour)
                val area = Imgproc.contourArea(contour)
                
                // Size filtering
                if (boundingRect.width >= 60 && boundingRect.height >= 60 &&
                    boundingRect.width <= 300 && boundingRect.height <= 300 &&
                    area > 2000) {
                    
                    // Aspect ratio filtering (faces are roughly square to slightly tall)
                    val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()
                    if (aspectRatio > 0.6 && aspectRatio < 1.4) {
                        
                        faceRegions.add(RectF(
                            boundingRect.x.toFloat(),
                            boundingRect.y.toFloat(),
                            (boundingRect.x + boundingRect.width).toFloat(),
                            (boundingRect.y + boundingRect.height).toFloat()
                        ))
                        Log.d(TAG, "Heatmap face: ${boundingRect.x},${boundingRect.y},${boundingRect.width}x${boundingRect.height}")
                    }
                }
            }
            
            // Cleanup
            morphMat.release()
            threshMat.release()
            thermalMat.release()
            hierarchy.release()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in contrast heatmap detection", e)
        }
        
        return faceRegions
    }
    
    private fun detectWithEdges(): List<RectF> {
        val faceRegions = mutableListOf<RectF>()
        
        try {
            // Fallback edge detection (simplified version)
            val edgesMat = Mat()
            Imgproc.Canny(grayMat, edgesMat, 50.0, 150.0)
            
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edgesMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            
            for (contour in contours) {
                val boundingRect = Imgproc.boundingRect(contour)
                val area = Imgproc.contourArea(contour)
                
                if (boundingRect.width >= 80 && boundingRect.height >= 80 &&
                    boundingRect.width <= 400 && boundingRect.height <= 400 &&
                    area > 5000) {
                    
                    val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()
                    if (aspectRatio > 0.7 && aspectRatio < 1.3) {
                        faceRegions.add(RectF(
                            boundingRect.x.toFloat(),
                            boundingRect.y.toFloat(),
                            (boundingRect.x + boundingRect.width).toFloat(),
                            (boundingRect.y + boundingRect.height).toFloat()
                        ))
                    }
                }
            }
            
            edgesMat.release()
            hierarchy.release()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in edge detection", e)
        }
        
        return faceRegions
    }
    
    private fun filterOverlappingFaces(faces: List<RectF>): List<RectF> {
        if (faces.size <= 1) return faces
        
        val filtered = mutableListOf<RectF>()
        val used = BooleanArray(faces.size)
        
        // Sort by area (largest first)
        val sortedIndices = faces.indices.sortedByDescending { 
            val face = faces[it]
            (face.right - face.left) * (face.bottom - face.top)
        }
        
        for (i in sortedIndices) {
            if (used[i]) continue
            
            val face1 = faces[i]
            var hasOverlap = false
            
            // Check if this face overlaps significantly with any already accepted face
            for (j in filtered.indices) {
                if (calculateOverlap(face1, filtered[j]) > 0.3) {
                    hasOverlap = true
                    break
                }
            }
            
            if (!hasOverlap) {
                filtered.add(face1)
                used[i] = true
            }
        }
        
        return filtered
    }
    
    private fun calculateOverlap(rect1: RectF, rect2: RectF): Float {
        val intersectLeft = maxOf(rect1.left, rect2.left)
        val intersectTop = maxOf(rect1.top, rect2.top)
        val intersectRight = minOf(rect1.right, rect2.right)
        val intersectBottom = minOf(rect1.bottom, rect2.bottom)
        
        if (intersectLeft >= intersectRight || intersectTop >= intersectBottom) {
            return 0f
        }
        
        val intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop)
        val area1 = (rect1.right - rect1.left) * (rect1.bottom - rect1.top)
        val area2 = (rect2.right - rect2.left) * (rect2.bottom - rect2.top)
        val unionArea = area1 + area2 - intersectArea
        
        return intersectArea / unionArea
    }
    
    fun getEyeRegions(faceRect: RectF): Pair<RectF, RectF>? {
        try {
            // Standard eye position estimation (Python OpenCV pattern)
            val faceWidth = faceRect.right - faceRect.left
            val faceHeight = faceRect.bottom - faceRect.top
            
            // Eyes are typically at 1/3 from top, 1/4 and 3/4 from left
            val eyeY = faceRect.top + faceHeight * 0.33f
            val eyeSize = faceWidth * 0.12f
            
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
        enhancedMat.release()
        heatmapMat.release()
        faces.release()
    }
}