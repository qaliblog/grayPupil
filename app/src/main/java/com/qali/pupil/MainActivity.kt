package com.qali.pupil


import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
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
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var processedFrameView: ProcessedFrameView
    private lateinit var cameraExecutor: ExecutorService
    private var isOpenCVLoaded = false

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
        processedFrameView = ProcessedFrameView(this)
        
        // Replace preview view with processed frame view
        val frameLayout = findViewById<android.widget.FrameLayout>(R.id.overlayContainer)
        frameLayout.removeAllViews()
        
        // Set proper layout parameters
        val layoutParams = android.widget.FrameLayout.LayoutParams(
            android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
            android.widget.FrameLayout.LayoutParams.MATCH_PARENT
        )
        processedFrameView.layoutParams = layoutParams
        frameLayout.addView(processedFrameView)
        
        // Hide the original preview view
        previewView.visibility = View.GONE
        
        Log.d(TAG, "ProcessedFrameView added to layout")
        
        // Create a simple test bitmap to verify display
        val testBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
        testBitmap.eraseColor(Color.GREEN)
        val canvas = Canvas(testBitmap)
        val paint = Paint().apply {
            color = Color.BLACK
            textSize = 40f
        }
        canvas.drawText("Display Working", 50f, 150f, paint)
        processedFrameView.updateFrame(testBitmap)
        Log.d(TAG, "Green test bitmap displayed")
        
        // Remove test bitmap after 3 seconds and show camera status
        processedFrameView.postDelayed({
            Log.d(TAG, "Removing test bitmap, checking camera status")
            val statusBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
            statusBitmap.eraseColor(Color.CYAN)
            val canvas = Canvas(statusBitmap)
            val paint = Paint().apply {
                color = Color.BLACK
                textSize = 24f
            }
            canvas.drawText("Waiting for camera...", 20f, 150f, paint)
            processedFrameView.updateFrame(statusBitmap)
        }, 3000)
        
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            Log.d(TAG, "Camera permission granted, starting camera")
            startCamera()
        } else {
            Log.w(TAG, "Camera permission not granted, requesting permission")
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
        Log.d(TAG, "Starting camera...")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            Log.d(TAG, "Camera provider ready")
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
                    Log.d(TAG, "Image analyzer set")
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )
                Log.d(TAG, "Camera bound successfully")
            } catch(e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
                // Show error on screen
                runOnUiThread {
                    val errorBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
                    errorBitmap.eraseColor(Color.RED)
                    val canvas = Canvas(errorBitmap)
                    val paint = Paint().apply {
                        color = Color.WHITE
                        textSize = 20f
                    }
                    canvas.drawText("Camera Error:", 20f, 100f, paint)
                    canvas.drawText(e.message ?: "Unknown error", 20f, 150f, paint)
                    processedFrameView.updateFrame(errorBitmap)
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private inner class ContourAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            try {
                Log.d(TAG, "Analyzing frame: ${imageProxy.width}x${imageProxy.height}")
                
                // Convert imageProxy to bitmap
                val bitmap = imageProxy.toBitmap()
                Log.d(TAG, "Original bitmap: ${bitmap.width}x${bitmap.height}")
                
                // For now, just show the original frame to test conversion
                runOnUiThread {
                    processedFrameView.updateFrame(bitmap)
                    Log.d(TAG, "Frame sent to display")
                }
                
                /*
                // TODO: Re-enable processing once basic display works
                if (!isOpenCVLoaded) {
                    Log.w(TAG, "OpenCV not loaded, showing original frame")
                    runOnUiThread {
                        processedFrameView.updateFrame(bitmap)
                    }
                    imageProxy.close()
                    return
                }
                
                // Apply contrast enhancement
                val enhancedBitmap = enhanceContrast(bitmap)
                Log.d(TAG, "Enhanced bitmap: ${enhancedBitmap.width}x${enhancedBitmap.height}")
                
                // Detect contours and draw them on the enhanced bitmap
                val contourBitmap = drawContoursOnBitmap(enhancedBitmap)
                Log.d(TAG, "Contour bitmap: ${contourBitmap.width}x${contourBitmap.height}")
                
                // Display the processed frame
                runOnUiThread {
                    processedFrameView.updateFrame(contourBitmap)
                }
                
                Log.d(TAG, "Processed frame with contrast enhancement and contours")
                */
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.message}", e)
                // Create a simple error bitmap
                val errorBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
                errorBitmap.eraseColor(Color.YELLOW)
                val canvas = Canvas(errorBitmap)
                val paint = Paint().apply {
                    color = Color.RED
                    textSize = 30f
                }
                canvas.drawText("Error: ${e.message}", 20f, 150f, paint)
                runOnUiThread {
                    processedFrameView.updateFrame(errorBitmap)
                }
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
                val step = maxOf(1, points.size / 10) // More frequent sampling
                for (i in points.indices step step) {
                    val point = points[i]
                    val squareSize = 16f // Larger squares for better visibility
                    
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
        
        Log.d(TAG, "Drew ${contours.size} contours on bitmap")
        
        return mutableBitmap
    }



    private fun ImageProxy.toBitmap(): Bitmap {
        return try {
            // Try YUV conversion first
            val yBuffer = planes[0].buffer // Y
            val vuBuffer = planes[2].buffer // VU

            val ySize = yBuffer.remaining()
            val vuSize = vuBuffer.remaining()

            val nv21 = ByteArray(ySize + vuSize)

            yBuffer.get(nv21, 0, ySize)
            vuBuffer.get(nv21, ySize, vuSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, this.width, this.height), 70, out)
            val imageBytes = out.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            
            Log.d(TAG, "YUV converted ImageProxy to bitmap: ${bitmap?.width}x${bitmap?.height}")
            bitmap ?: throw IllegalStateException("YUV conversion returned null")
        } catch (e: Exception) {
            Log.w(TAG, "YUV conversion failed, trying simple conversion: ${e.message}")
            
            // Fallback: simple conversion (may not work for all formats)
            val buffer = planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            
            Log.d(TAG, "Simple converted ImageProxy to bitmap: ${bitmap?.width}x${bitmap?.height}")
            bitmap ?: createDummyBitmap()
        }
    }
    
    private fun createDummyBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.BLUE)
        val canvas = Canvas(bitmap)
        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 50f
        }
        canvas.drawText("Camera Feed", 50f, 240f, paint)
        Log.d(TAG, "Created dummy bitmap")
        return bitmap
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
        
        // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with stronger enhancement
        val clahe = Imgproc.createCLAHE(4.0, OpenCVSize(8.0, 8.0))
        val enhancedMat = Mat()
        clahe.apply(grayMat, enhancedMat)
        
        // Convert enhanced grayscale back to RGB for display
        val rgbMat = Mat()
        Imgproc.cvtColor(enhancedMat, rgbMat, Imgproc.COLOR_GRAY2RGB)
        
        // Convert back to bitmap
        val resultBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbMat, resultBitmap)
        
        rgbMat.release()
        
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
        val clahe = Imgproc.createCLAHE(3.0, OpenCVSize(8.0, 8.0))
        val enhancedMat = Mat()
        clahe.apply(grayMat, enhancedMat)
        
        // Apply Gaussian blur to reduce noise
        val blurredMat = Mat()
        Imgproc.GaussianBlur(enhancedMat, blurredMat, OpenCVSize(3.0, 3.0), 0.0)
        
        // Apply Canny edge detection with adjusted thresholds
        val edgesMat = Mat()
        Imgproc.Canny(blurredMat, edgesMat, 30.0, 100.0)
        
        // Find contours
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edgesMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        // Filter contours by area - focus on significant contours
        val filteredContours = contours.filter { contour ->
            val area = Imgproc.contourArea(contour)
            area > 20.0 // Lower threshold to catch more contours
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

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG, "Camera permission granted after request")
                startCamera()
            } else {
                Log.e(TAG, "Camera permission denied")
                val errorBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
                errorBitmap.eraseColor(Color.RED)
                val canvas = Canvas(errorBitmap)
                val paint = Paint().apply {
                    color = Color.WHITE
                    textSize = 20f
                }
                canvas.drawText("Camera permission", 20f, 100f, paint)
                canvas.drawText("required", 20f, 150f, paint)
                processedFrameView.updateFrame(errorBitmap)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}