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

// OpenCV imports temporarily removed - will re-add with proper Android OpenCV
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var processedFrameView: ProcessedFrameView
    private lateinit var cameraExecutor: ExecutorService
    private var frameCount = 0
    // OpenCV temporarily disabled

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
            Log.d(TAG, "Test period over, checking if frames are coming...")
            if (frameCount == 0) {
                Log.w(TAG, "No frames received yet! Camera analyzer not being called")
                val errorBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
                errorBitmap.eraseColor(Color.YELLOW)
                val canvas = Canvas(errorBitmap)
                val paint = Paint().apply {
                    color = Color.RED
                    textSize = 30f
                }
                canvas.drawText("NO CAMERA", 50f, 150f, paint)
                canvas.drawText("FRAMES!", 50f, 200f, paint)
                processedFrameView.updateFrame(errorBitmap)
            } else {
                Log.d(TAG, "Frames are coming through: $frameCount received")
            }
        }, 5000)
        
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            Log.d(TAG, "Camera permission granted, starting camera")
            startCamera()
        } else {
            Log.w(TAG, "Camera permission not granted, requesting permission")
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        }
    }

    // onResume OpenCV initialization temporarily removed

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
                frameCount++
                Log.d(TAG, "ANALYZER CALLED - Frame #$frameCount: ${imageProxy.width}x${imageProxy.height}")
                
                // Convert imageProxy to bitmap
                val bitmap = imageProxy.toBitmap()
                Log.d(TAG, "Bitmap created: ${bitmap.width}x${bitmap.height}")
                
                // For now, just show the original frame to test conversion
                runOnUiThread {
                    processedFrameView.updateFrame(bitmap)
                    Log.d(TAG, "Frame #$frameCount sent to display")
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

    // drawContoursOnBitmap temporarily removed - will re-add with OpenCV



    private fun ImageProxy.toBitmap(): Bitmap {
        Log.d(TAG, "Converting ImageProxy format: ${format}, size: ${width}x${height}, planes: ${planes.size}")
        
        return try {
            // Create test pattern with frame counter
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.eraseColor(Color.WHITE)
            
            val canvas = Canvas(bitmap)
            val paint = Paint().apply {
                color = Color.BLACK
                textSize = 40f
                isAntiAlias = true
            }
            
            // Draw frame information
            canvas.drawText("FRAME #$frameCount", 50f, 100f, paint)
            canvas.drawText("Size: ${width}x${height}", 50f, 150f, paint)
            canvas.drawText("Format: $format", 50f, 200f, paint)
            canvas.drawText("Planes: ${planes.size}", 50f, 250f, paint)
            
            // Draw a moving pattern based on frame count
            val patternPaint = Paint().apply {
                color = Color.RED
                style = Paint.Style.FILL
            }
            val x = (frameCount * 5) % width
            canvas.drawCircle(x.toFloat(), 300f, 20f, patternPaint)
            
            Log.d(TAG, "Test pattern #$frameCount created: ${bitmap.width}x${bitmap.height}")
            bitmap

        } catch (e: Exception) {
            Log.e(TAG, "Even test pattern failed: ${e.message}", e)
            createTestPattern()
        }
    }
    
    private fun createTestPattern(): Bitmap {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.BLUE)
        val canvas = Canvas(bitmap)
        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 24f
        }
        canvas.drawText("Camera conversion", 50f, 200f, paint)
        canvas.drawText("failed - check logs", 50f, 240f, paint)
        Log.d(TAG, "Created blue test pattern - camera conversion failed")
        return bitmap
    }

    

    // OpenCV processing functions temporarily removed
    // Will re-add with proper Android OpenCV dependency

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