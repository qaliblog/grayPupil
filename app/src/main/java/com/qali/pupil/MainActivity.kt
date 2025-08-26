package com.qali.pupil


import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat

import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Size as OpenCVSize
import org.opencv.imgproc.Imgproc

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private var isOpenCVLoaded = false
    private var processedFrame: Bitmap? = null

    // OpenCV loader callback
    private val openCVLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.d(TAG, "OpenCV loaded successfully")
                    isOpenCVLoaded = true
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    companion object {
        private const val TAG = "ContourDetection"
        private const val CAMERA_PERMISSION_CODE = 101
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        previewView = findViewById(R.id.previewView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        }
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, openCVLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            openCVLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    private fun allPermissionsGranted() = 
        ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ContourAnalyzer())
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )
            } catch(e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private inner class ContourAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (!isOpenCVLoaded) {
                imageProxy.close()
                return
            }

            try {
                // Convert imageProxy to bitmap
                val bitmap = imageProxy.toBitmap()
                
                // Apply contrast enhancement
                val enhancedBitmap = enhanceContrast(bitmap)
                
                // Detect contours and draw them on the enhanced bitmap
                val contourBitmap = drawContoursOnBitmap(enhancedBitmap)
                
                // Store the processed frame
                processedFrame = contourBitmap
                
                Log.d(TAG, "Processed frame with contrast enhancement and contours")
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.message}")
            } finally {
                imageProxy.close()
            }
        }
    }

    private fun drawContoursOnBitmap(bitmap: Bitmap): Bitmap {
        if (!isOpenCVLoaded) return bitmap
        
        // Detect contours
        val contours = detectImageContours(bitmap)
        
        // Create a mutable copy of the bitmap to draw on
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        
        // Paint for green squares
        val paint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        
        // Draw green squares at contour points
        for (contour in contours) {
            val points = contour.toArray()
            if (points.isNotEmpty()) {
                val step = maxOf(1, points.size / 20) // Sample points
                for (i in points.indices step step) {
                    val point = points[i]
                    val squareSize = 12f
                    
                    canvas.drawRect(
                        point.x.toFloat() - squareSize / 2,
                        point.y.toFloat() - squareSize / 2,
                        point.x.toFloat() + squareSize / 2,
                        point.y.toFloat() + squareSize / 2,
                        paint
                    )
                }
            }
        }
        
        return mutableBitmap
    }



    private fun ImageProxy.toBitmap(): Bitmap {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    

    private fun enhanceContrast(bitmap: Bitmap): Bitmap {
        if (!isOpenCVLoaded) {
            Log.w(TAG, "OpenCV not loaded, skipping contrast enhancement")
            return bitmap
        }
        
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        
        // Convert to grayscale for contrast enhancement
        val grayMat = Mat()
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)
        
        // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        val clahe = Imgproc.createCLAHE(3.0, OpenCVSize(8.0, 8.0))
        val enhancedMat = Mat()
        clahe.apply(grayMat, enhancedMat)
        
        // Convert back to bitmap
        val resultBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(enhancedMat, resultBitmap)
        
        mat.release()
        grayMat.release()
        enhancedMat.release()
        
        return resultBitmap
    }

    private fun detectImageContours(bitmap: Bitmap): List<MatOfPoint> {
        if (!isOpenCVLoaded) {
            Log.w(TAG, "OpenCV not loaded, skipping contour detection")
            return emptyList()
        }
        
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        
        // Convert to grayscale
        val grayMat = Mat()
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)
        
        // Apply contrast enhancement
        val clahe = Imgproc.createCLAHE(2.0, OpenCVSize(8.0, 8.0))
        val enhancedMat = Mat()
        clahe.apply(grayMat, enhancedMat)
        
        // Apply Gaussian blur to reduce noise
        val blurredMat = Mat()
        Imgproc.GaussianBlur(enhancedMat, blurredMat, OpenCVSize(5.0, 5.0), 0.0)
        
        // Apply Canny edge detection
        val edgesMat = Mat()
        Imgproc.Canny(blurredMat, edgesMat, 50.0, 150.0)
        
        // Find contours
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edgesMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        // Filter contours by area - focus on larger face contours
        val filteredContours = contours.filter { contour ->
            val area = Imgproc.contourArea(contour)
            area > 50.0 // Minimum area for face features (reduced for 64x64 images)
        }
        
        Log.d(TAG, "Found ${contours.size} total contours, ${filteredContours.size} after filtering")
        
        mat.release()
        grayMat.release()
        enhancedMat.release()
        blurredMat.release()
        edgesMat.release()
        hierarchy.release()
        
        return filteredContours
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}