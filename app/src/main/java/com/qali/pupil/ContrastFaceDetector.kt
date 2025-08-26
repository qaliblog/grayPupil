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

class ContrastFaceDetector(private val context: Context) {
    companion object {
        private const val TAG = "ContrastFaceDetector"
        
        // Standard Python OpenCV parameters
        private const val SCALE_FACTOR = 1.1
        private const val MIN_NEIGHBORS = 4
        private const val MIN_SIZE_FACTOR = 0.1  // Min face size as fraction of image
    }

    // OpenCV Mats (following Python variable naming)
    private val gray = Mat()           // gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    private val faces = MatOfRect()    // faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    private var faceCascade: CascadeClassifier? = null  // face_cascade = cv2.CascadeClassifier(...)

    init {
        loadCascadeClassifier()
    }

    // Step 1: Load Haar cascade (Python: face_cascade = cv2.CascadeClassifier(...))
    private fun loadCascadeClassifier() {
        try {
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
            
            faceCascade = CascadeClassifier(cascadeFile.absolutePath)
            
            if (faceCascade?.empty() == true) {
                Log.e(TAG, "Failed to load cascade classifier")
                faceCascade = null
            } else {
                Log.d(TAG, "Cascade classifier loaded successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading cascade", e)
            faceCascade = null
        }
    }

    // Main detection method following exact Python OpenCV flow
    fun detectFaces(bitmap: Bitmap): List<RectF> {
        try {
            // Python: img = cv2.imread('image.jpg')
            val img = Mat(bitmap.height, bitmap.width, CvType.CV_8UC3)
            Utils.bitmapToMat(bitmap, img)
            
            // Python: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_RGB2GRAY)
            
            // Optional: Apply histogram equalization for better contrast (common Python enhancement)
            // Python: gray = cv2.equalizeHist(gray)
            Imgproc.equalizeHist(gray, gray)
            
            // Python: faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            val detectedFaces = detectMultiScale()
            
            // Python: for (x, y, w, h) in faces: ...
            val faceRectangles = convertToRectF(detectedFaces)
            
            Log.d(TAG, "Detected ${faceRectangles.size} faces using Python OpenCV pattern")
            return faceRectangles
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in face detection", e)
            return emptyList()
        }
    }
    
    // Python: faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    private fun detectMultiScale(): Array<Rect> {
        val classifier = faceCascade
        if (classifier == null) {
            Log.w(TAG, "Cascade classifier not loaded, using fallback detection")
            return detectWithContrastFallback()
        }
        
        try {
            // Calculate minimum face size (Python common practice)
            val minSize = Size(
                (gray.cols() * MIN_SIZE_FACTOR).toDouble(),
                (gray.rows() * MIN_SIZE_FACTOR).toDouble()
            )
            
            // Python: face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            classifier.detectMultiScale(
                gray,                    // Input image
                faces,                   // Output rectangles
                SCALE_FACTOR,           // Scale factor (1.1)
                MIN_NEIGHBORS,          // Min neighbors (4)
                0,                      // Flags
                minSize,                // Minimum size
                Size()                  // Maximum size (empty = no limit)
            )
            
            val detectedFaces = faces.toArray()
            Log.d(TAG, "Haar cascade detected ${detectedFaces.size} faces")
            
            return detectedFaces
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in detectMultiScale", e)
            return emptyArray()
        }
    }
    
    // Fallback method using contrast when Haar cascade is not available
    private fun detectWithContrastFallback(): Array<Rect> {
        try {
            Log.d(TAG, "Using contrast-based fallback detection")
            
            // Apply additional contrast enhancement
            val enhanced = Mat()
            gray.convertTo(enhanced, CvType.CV_8UC1, 1.2, 10.0)
            
            // Find edges using Canny
            val edges = Mat()
            Imgproc.Canny(enhanced, edges, 50.0, 150.0)
            
            // Find contours
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            
            val faceRects = mutableListOf<Rect>()
            
            for (contour in contours) {
                val boundingRect = Imgproc.boundingRect(contour)
                val area = Imgproc.contourArea(contour)
                
                // Filter for face-like regions
                if (boundingRect.width >= 60 && boundingRect.height >= 60 &&
                    boundingRect.width <= 300 && boundingRect.height <= 300 &&
                    area > 2000) {
                    
                    val aspectRatio = boundingRect.width.toDouble() / boundingRect.height.toDouble()
                    if (aspectRatio > 0.6 && aspectRatio < 1.4) {
                        faceRects.add(boundingRect)
                    }
                }
            }
            
            // Cleanup
            enhanced.release()
            edges.release()
            hierarchy.release()
            
            Log.d(TAG, "Contrast fallback found ${faceRects.size} face candidates")
            return faceRects.toTypedArray()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in contrast fallback", e)
            return emptyArray()
        }
    }
    
    // Python: for (x, y, w, h) in faces: cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    private fun convertToRectF(detectedFaces: Array<Rect>): List<RectF> {
        val rectangles = mutableListOf<RectF>()
        
        for (face in detectedFaces) {
            // Convert OpenCV Rect to Android RectF
            // Python: (x, y, w, h) -> Android: (left, top, right, bottom)
            val rectF = RectF(
                face.x.toFloat(),                    // x -> left
                face.y.toFloat(),                    // y -> top  
                (face.x + face.width).toFloat(),     // x + w -> right
                (face.y + face.height).toFloat()     // y + h -> bottom
            )
            rectangles.add(rectF)
            
            Log.d(TAG, "Face rectangle: x=${face.x}, y=${face.y}, w=${face.width}, h=${face.height}")
        }
        
        return rectangles
    }
    
    // Standard eye region estimation (common in Python OpenCV tutorials)
    fun getEyeRegions(faceRect: RectF): Pair<RectF, RectF>? {
        try {
            val faceWidth = faceRect.right - faceRect.left
            val faceHeight = faceRect.bottom - faceRect.top
            
            // Standard proportions from Python OpenCV examples
            val eyeY = faceRect.top + faceHeight * 0.35f
            val eyeSize = faceWidth * 0.15f
            
            val leftEyeX = faceRect.left + faceWidth * 0.3f
            val rightEyeX = faceRect.left + faceWidth * 0.7f
            
            val leftEyeRect = RectF(
                leftEyeX - eyeSize/2,
                eyeY - eyeSize/2,
                leftEyeX + eyeSize/2,
                eyeY + eyeSize/2
            )
            
            val rightEyeRect = RectF(
                rightEyeX - eyeSize/2,
                eyeY - eyeSize/2,
                rightEyeX + eyeSize/2,
                eyeY + eyeSize/2
            )
            
            return Pair(leftEyeRect, rightEyeRect)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error getting eye regions", e)
            return null
        }
    }
    
    fun release() {
        gray.release()
        faces.release()
    }
}