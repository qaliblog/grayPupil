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
import android.view.Surface
import android.view.View
import android.widget.SeekBar
import android.widget.TextView
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
    private var frameCount = 0
    
    // Threshold control variables
    private var claheClipLimit = 15.0
    private var cannyLowThreshold = 50.0
    private var cannyHighThreshold = 150.0
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
        Log.d(TAG, "Camera executor created")
        
        // Setup threshold control SeekBars
        setupThresholdControls()

        if (allPermissionsGranted()) {
            Log.d(TAG, "Camera permission granted, starting camera")
            startCamera()
        } else {
            Log.w(TAG, "Camera permission not granted, requesting permission")
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        }
    }
    
    private fun setupThresholdControls() {
        val thresholdSeekBar = findViewById<SeekBar>(R.id.thresholdSeekBar)
        val thresholdLabel = findViewById<TextView>(R.id.thresholdLabel)
        val cannyLowSeekBar = findViewById<SeekBar>(R.id.cannyLowSeekBar)
        val cannyHighSeekBar = findViewById<SeekBar>(R.id.cannyHighSeekBar)
        val cannyLabel = findViewById<TextView>(R.id.cannyLabel)
        
        // CLAHE threshold control
        thresholdSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                claheClipLimit = progress.toDouble().coerceAtLeast(1.0)
                thresholdLabel.text = "Contrast Threshold: $claheClipLimit"
                Log.d(TAG, "CLAHE clip limit updated to: $claheClipLimit")
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        // Canny low threshold control
        cannyLowSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                cannyLowThreshold = progress.toDouble()
                updateCannyLabel(cannyLabel)
                Log.d(TAG, "Canny low threshold updated to: $cannyLowThreshold")
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        // Canny high threshold control
        cannyHighSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                cannyHighThreshold = progress.toDouble()
                updateCannyLabel(cannyLabel)
                Log.d(TAG, "Canny high threshold updated to: $cannyHighThreshold")
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        Log.d(TAG, "Threshold controls setup complete")
    }
    
    private fun updateCannyLabel(cannyLabel: TextView) {
        cannyLabel.text = "Canny Low: ${cannyLowThreshold.toInt()} | High: ${cannyHighThreshold.toInt()}"
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
            try {
                Log.d(TAG, "Camera provider ready")
                val cameraProvider = cameraProviderFuture.get()
                
                // Check available cameras
                val availableCameras = cameraProvider.availableCameraInfos
                Log.d(TAG, "Available cameras: ${availableCameras.size}")
                
                // Use front camera for face detection
                Log.d(TAG, "Using front camera...")
                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                
                // Force portrait mode with target rotation
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(480, 640)) // Portrait: height > width
                    .setTargetRotation(Surface.ROTATION_0) // Force portrait rotation
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, ContourAnalyzer())
                        Log.d(TAG, "Image analyzer set on executor")
                    }

                // Bind to lifecycle - just image analysis
                cameraProvider.unbindAll()
                Log.d(TAG, "Unbound all previous cameras")
                
                val camera = cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    imageAnalysis  // Only bind image analysis, no preview
                )
                Log.d(TAG, "Camera bound successfully to lifecycle")
                Log.d(TAG, "Camera info: ${camera.cameraInfo}")
                
            } catch(e: Exception) {
                Log.e(TAG, "Camera setup failed", e)
                // Show error on screen
                runOnUiThread {
                    val errorBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
                    errorBitmap.eraseColor(Color.RED)
                    val canvas = Canvas(errorBitmap)
                    val paint = Paint().apply {
                        color = Color.WHITE
                        textSize = 16f
                    }
                    canvas.drawText("Camera Setup Failed:", 20f, 80f, paint)
                    canvas.drawText(e.javaClass.simpleName, 20f, 110f, paint)
                    canvas.drawText(e.message?.take(40) ?: "Unknown", 20f, 140f, paint)
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
                
                // Apply contrast enhancement and contour detection
                val enhancedBitmap = enhanceContrast(bitmap)
                val contourBitmap = drawContoursOnBitmap(enhancedBitmap)
                
                runOnUiThread {
                    processedFrameView.updateFrame(contourBitmap)
                    Log.d(TAG, "Frame #$frameCount with OpenCV processing sent to display")
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
        Log.d(TAG, "Camera rotation degrees: ${imageInfo.rotationDegrees}")
        
        return try {
            // Proper YUV_420_888 to grayscale conversion
            val yPlane = planes[0]
            val yBuffer = yPlane.buffer
            val ySize = yBuffer.remaining()
            val yPixelStride = yPlane.pixelStride
            val yRowStride = yPlane.rowStride
            
            Log.d(TAG, "YUV details - PixelStride: $yPixelStride, RowStride: $yRowStride, BufferSize: $ySize")
            Log.d(TAG, "Image dimensions: ${width}x${height}")
            
            // Simple YUV conversion + MASSIVE rotation test
            val pixels = IntArray(width * height)
            var index = 0
            
            for (row in 0 until height) {
                for (col in 0 until width) {
                    val bufferIndex = row * yRowStride + col * yPixelStride
                    if (bufferIndex < ySize) {
                        val y = yBuffer.get(bufferIndex).toInt() and 0xFF
                        pixels[index] = (0xFF shl 24) or (y shl 16) or (y shl 8) or y
                    } else {
                        pixels[index] = 0xFF000000.toInt()
                    }
                    index++
                }
            }
            
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            
            // EXTREME rotation test - try 180 degrees!
            val matrix = Matrix().apply {
                Log.d(TAG, "TRYING 180° ROTATION - FLIP UPSIDE DOWN")
                postRotate(180f) // Maybe we need to flip it completely?
            }
            
            val finalBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
            
            Log.d(TAG, "Final bitmap: ${finalBitmap.width}x${finalBitmap.height}")
            finalBitmap

        } catch (e: Exception) {
            Log.e(TAG, "Camera conversion failed: ${e.message}", e)
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
        
        // Apply dynamic CLAHE for contrast enhancement
        val clahe = Imgproc.createCLAHE(claheClipLimit, OpenCVSize(8.0, 8.0))
        val enhancedMat = Mat()
        clahe.apply(grayMat, enhancedMat)
        
        // Additional histogram equalization for maximum contrast
        val equalizedMat = Mat()
        Imgproc.equalizeHist(enhancedMat, equalizedMat)
        
        // Apply morphological operations to enhance edges
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, OpenCVSize(3.0, 3.0))
        val morphMat = Mat()
        Imgproc.morphologyEx(equalizedMat, morphMat, Imgproc.MORPH_GRADIENT, kernel)
        
        // Convert enhanced grayscale back to RGB for display
        val rgbMat = Mat()
        Imgproc.cvtColor(morphMat, rgbMat, Imgproc.COLOR_GRAY2RGB)
        
        // Convert back to bitmap
        val resultBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbMat, resultBitmap)
        
        rgbMat.release()
        mat.release()
        grayMat.release()
        enhancedMat.release()
        equalizedMat.release()
        morphMat.release()
        kernel.release()
        
        Log.d(TAG, "Contrast enhanced successfully")
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
        
        // Apply dynamic contrast enhancement for contour detection
        val clahe = Imgproc.createCLAHE(claheClipLimit, OpenCVSize(8.0, 8.0))
        val enhancedMat = Mat()
        clahe.apply(grayMat, enhancedMat)
        
        // Additional histogram equalization for maximum edge visibility
        val equalizedMat = Mat()
        Imgproc.equalizeHist(enhancedMat, equalizedMat)
        
        // Apply morphological gradient to enhance edge contrast
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, OpenCVSize(3.0, 3.0))
        val morphMat = Mat()
        Imgproc.morphologyEx(equalizedMat, morphMat, Imgproc.MORPH_GRADIENT, kernel)
        
        // Apply stronger Gaussian blur to focus on larger features
        val blurredMat = Mat()
        Imgproc.GaussianBlur(morphMat, blurredMat, OpenCVSize(7.0, 7.0), 0.0)
        
        // Apply Canny edge detection with dynamic thresholds
        val edgesMat = Mat()
        Imgproc.Canny(blurredMat, edgesMat, cannyLowThreshold, cannyHighThreshold)
        
        // Find contours
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edgesMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        // Filter contours by area - more sensitive for face detection
        val imageArea = mat.width() * mat.height()
        val minFaceArea = imageArea * 0.02  // At least 2% of image (more sensitive)
        val maxFaceArea = imageArea * 0.8   // At most 80% of image
        
        val filteredContours = contours.filter { contour ->
            val area = Imgproc.contourArea(contour)
            area >= minFaceArea && area <= maxFaceArea
        }
        
        Log.d(TAG, "Image area: $imageArea, Face area range: $minFaceArea - $maxFaceArea")
        
        Log.d(TAG, "Found ${contours.size} total contours, ${filteredContours.size} after filtering")
        
        mat.release()
        grayMat.release()
        enhancedMat.release()
        equalizedMat.release()
        morphMat.release()
        kernel.release()
        blurredMat.release()
        edgesMat.release()
        hierarchy.release()
        
        return filteredContours
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
        
        // Draw green squares at contour points - larger squares, less frequent for face contours
        for (contour in contours) {
            val points = contour.toArray()
            if (points.isNotEmpty()) {
                val step = maxOf(1, points.size / 20) // Less frequent sampling for cleaner look
                for (i in points.indices step step) {
                    val point = points[i]
                    val squareSize = 24f // Larger squares for face-sized contours
                    
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