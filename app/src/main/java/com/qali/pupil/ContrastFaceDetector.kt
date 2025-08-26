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
        
        // Exact Python OpenCV standard parameters
        private const val SCALE_FACTOR = 1.1
        private const val MIN_NEIGHBORS = 4
        private const val MIN_SIZE_FACTOR = 0.1
    }

    // Python OpenCV variables (exact naming convention)
    private val gray = Mat()           // gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    private val faces = MatOfRect()    // faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    private var face_cascade: CascadeClassifier? = null  // face_cascade = cv2.CascadeClassifier(...)

    init {
        loadCascade()
    }

    // Python: face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    private fun loadCascade() {
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
            
            face_cascade = CascadeClassifier(cascadeFile.absolutePath)
            
            if (face_cascade?.empty() == true) {
                Log.e(TAG, "Failed to load cascade classifier")
                face_cascade = null
            } else {
                Log.d(TAG, "Haar cascade loaded successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading cascade", e)
            face_cascade = null
        }
    }

    // EXACT Python OpenCV face detection function
    fun detectFaces(bitmap: Bitmap): List<RectF> {
        /*
        Python equivalent:
        import cv2
        
        def detect_faces(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append((x, y, x+w, y+h))
            return face_list
        */
        
        try {
            // Python: img = cv2.imread('image.jpg')
            val img = Mat(bitmap.height, bitmap.width, CvType.CV_8UC3)
            Utils.bitmapToMat(bitmap, img)
            
            // Python: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_RGB2GRAY)
            
            // Python: faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            val classifier = face_cascade
            if (classifier == null) {
                Log.e(TAG, "Haar cascade not loaded")
                return emptyList()
            }
            
            // Calculate minimum face size (standard Python practice)
            val minSize = Size(
                (gray.cols() * MIN_SIZE_FACTOR).toDouble(),
                (gray.rows() * MIN_SIZE_FACTOR).toDouble()
            )
            
            // Python: face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            classifier.detectMultiScale(
                gray,               // input image
                faces,              // output detections
                SCALE_FACTOR,       // 1.1
                MIN_NEIGHBORS,      // 4
                0,                  // flags (0 = default)
                minSize,            // minimum size
                Size()              // maximum size (empty = no limit)
            )
            
            // Python: for (x, y, w, h) in faces:
            val detectedFaces = faces.toArray()
            val faceList = mutableListOf<RectF>()
            
            for (face in detectedFaces) {
                // Python: (x, y, w, h) -> convert to rectangle
                val rectF = RectF(
                    face.x.toFloat(),                    // x
                    face.y.toFloat(),                    // y  
                    (face.x + face.width).toFloat(),     // x + w
                    (face.y + face.height).toFloat()     // y + h
                )
                faceList.add(rectF)
                
                Log.d(TAG, "Detected face: x=${face.x}, y=${face.y}, w=${face.width}, h=${face.height}")
            }
            
            Log.d(TAG, "Total faces detected: ${faceList.size}")
            return faceList
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in face detection", e)
            return emptyList()
        }
    }
    
    // Standard eye detection (common in Python OpenCV examples)
    fun getEyeRegions(faceRect: RectF): Pair<RectF, RectF>? {
        /*
        Python equivalent:
        def get_eye_regions(face_rect):
            x, y, w, h = face_rect
            eye_w = int(w * 0.15)
            eye_h = int(h * 0.15)
            
            left_eye_x = x + int(w * 0.3)
            right_eye_x = x + int(w * 0.7)
            eye_y = y + int(h * 0.35)
            
            left_eye = (left_eye_x - eye_w//2, eye_y - eye_h//2, left_eye_x + eye_w//2, eye_y + eye_h//2)
            right_eye = (right_eye_x - eye_w//2, eye_y - eye_h//2, right_eye_x + eye_w//2, eye_y + eye_h//2)
            
            return left_eye, right_eye
        */
        
        try {
            val w = faceRect.right - faceRect.left
            val h = faceRect.bottom - faceRect.top
            
            val eye_w = w * 0.15f
            val eye_h = h * 0.15f
            
            val left_eye_x = faceRect.left + w * 0.3f
            val right_eye_x = faceRect.left + w * 0.7f
            val eye_y = faceRect.top + h * 0.35f
            
            val leftEyeRect = RectF(
                left_eye_x - eye_w/2,
                eye_y - eye_h/2,
                left_eye_x + eye_w/2,
                eye_y + eye_h/2
            )
            
            val rightEyeRect = RectF(
                right_eye_x - eye_w/2,
                eye_y - eye_h/2,
                right_eye_x + eye_w/2,
                eye_y + eye_h/2
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