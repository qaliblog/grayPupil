package com.qali.pupil

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.RectF
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
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: Interpreter
    private lateinit var overlayView: GazeOverlayView
    
    // Contrast-based face detection components
    private lateinit var contrastFaceDetector: ContrastFaceDetector
    private var isOpenCVInitialized = false
    private var frameCount = 0

    // Model parameters
    private val INPUT_SIZE = 64
    private val GAZE_HISTORY_SIZE = 5
    private val gazeHistory = mutableListOf<Pair<Float, Float>>()

    companion object {
        private const val TAG = "EyeTracking"
        private const val CAMERA_PERMISSION_CODE = 101
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        Log.d(TAG, "=== MainActivity onCreate started ===")
        
        previewView = findViewById(R.id.previewView)
        overlayView = GazeOverlayView(this)
        findViewById<android.widget.FrameLayout>(R.id.overlayContainer).addView(overlayView)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        Log.d(TAG, "Views initialized, checking permissions...")
        
        // Initialize OpenCV first, then start camera
        initializeOpenCV()

        if (allPermissionsGranted()) {
            Log.d(TAG, "Camera permissions granted")
        } else {
            Log.d(TAG, "Requesting camera permissions")
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        }
    }
    
    private fun initializeOpenCV() {
        Log.d(TAG, "=== Starting OpenCV initialization ===")
        com.qali.pupil.OpenCVLoader.initializeOpenCV(this) {
            runOnUiThread {
                Log.d(TAG, "OpenCV initialization completed successfully")
                isOpenCVInitialized = true
                contrastFaceDetector = ContrastFaceDetector()
                
                try {
                    tflite = Interpreter(loadModelFile("gaze_model.tflite"))
                    Log.d(TAG, "TensorFlow Lite model loaded successfully")
                    if (allPermissionsGranted()) {
                        startCamera()
                    }
                } catch (e: IOException) {
                    Log.e(TAG, "Error loading TensorFlow Lite model - continuing without gaze estimation", e)
                    // Continue without TensorFlow model for now
                    if (allPermissionsGranted()) {
                        startCamera()
                    }
                }
            }
        }
    }

    private fun allPermissionsGranted() = 
        ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        Log.d(TAG, "=== Starting camera setup ===")
        
        if (!isOpenCVInitialized) {
            Log.w(TAG, "OpenCV not initialized yet, waiting...")
            return
        }
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                Log.d(TAG, "Camera provider obtained successfully")
                
                val preview = Preview.Builder()
                    .setTargetResolution(Size(640, 480))
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                        Log.d(TAG, "Preview surface provider set")
                    }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, ContrastFaceAnalyzer())
                        Log.d(TAG, "Image analyzer set")
                    }

                // Try front camera first, then back camera
                var cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                    Log.d(TAG, "Front camera bound successfully")
                } catch(e: Exception) {
                    Log.w(TAG, "Front camera failed, trying back camera", e)
                    try {
                        cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            this,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )
                        Log.d(TAG, "Back camera bound successfully")
                    } catch(e2: Exception) {
                        Log.e(TAG, "Both cameras failed to bind", e2)
                        showTestPattern()
                    }
                }
            } catch(e: Exception) {
                Log.e(TAG, "Camera setup failed completely", e)
                showTestPattern()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun showTestPattern() {
        Log.d(TAG, "Showing test pattern instead of camera")
        runOnUiThread {
            // Create a test pattern to verify the detection pipeline works
            val testFaces = listOf(
                RectF(100f, 150f, 300f, 350f), // Center face
                RectF(400f, 100f, 550f, 250f)  // Side face
            )
            overlayView.updateFaceRegions(testFaces, 640, 480) // Use standard camera size
            overlayView.updateGazePoint(0.5f, 0.5f) // Center gaze point
        }
    }

    private inner class ContrastFaceAnalyzer : ImageAnalysis.Analyzer {
        private var lastBitmapHash = 0
        
        override fun analyze(imageProxy: ImageProxy) {
            frameCount++
            
            if (!isOpenCVInitialized) {
                Log.w(TAG, "Frame $frameCount: OpenCV not initialized, skipping")
                imageProxy.close()
                return
            }
            
            try {
                Log.d(TAG, "=== Frame $frameCount Analysis ===")
                Log.d(TAG, "ImageProxy format: ${imageProxy.format}")
                Log.d(TAG, "ImageProxy size: ${imageProxy.width}x${imageProxy.height}")
                Log.d(TAG, "ImageProxy planes: ${imageProxy.planes.size}")
                
                val bitmap = imageProxy.toBitmap()
                if (bitmap == null) {
                    Log.e(TAG, "Failed to convert ImageProxy to bitmap")
                    imageProxy.close()
                    return
                }
                
                Log.d(TAG, "Bitmap created: ${bitmap.width}x${bitmap.height}, config: ${bitmap.config}")
                
                // Analyze bitmap content to see if we're getting valid camera data
                val pixels = IntArray(100) // Sample first 100 pixels
                bitmap.getPixels(pixels, 0, 10, 0, 0, 10, 10)
                val avgColor = pixels.average().toInt()
                val variance = pixels.map { (it - avgColor) * (it - avgColor) }.average()
                val bitmapHash = pixels.contentHashCode()
                
                Log.d(TAG, "Bitmap sample - avgColor: $avgColor, variance: $variance")
                
                // Check if camera is providing changing data
                if (bitmapHash == lastBitmapHash) {
                    Log.w(TAG, "WARNING: Camera data appears static/frozen!")
                } else {
                    Log.d(TAG, "Camera data is changing (good)")
                }
                lastBitmapHash = bitmapHash
                
                // Try simple detection first with even more permissive settings
                var faces = detectFacesBasic(bitmap)
                Log.d(TAG, "Basic detection found ${faces.size} faces")
                
                if (faces.isEmpty()) {
                    faces = contrastFaceDetector.detectFacesSimple(bitmap)
                    Log.d(TAG, "Simple detection found ${faces.size} faces")
                }
                
                if (faces.isEmpty()) {
                    faces = contrastFaceDetector.detectFaces(bitmap)
                    Log.d(TAG, "Advanced detection found ${faces.size} faces")
                }
                
                // Update overlay with detected faces for visualization
                runOnUiThread {
                    overlayView.updateFaceRegions(faces, bitmap.width, bitmap.height)
                }
                
                if (faces.isNotEmpty()) {
                    Log.d(TAG, "Processing face: ${faces[0]}")
                    Log.d(TAG, "Camera frame size: ${bitmap.width}x${bitmap.height}")
                    Log.d(TAG, "Face in camera coords: left=${faces[0].left}, top=${faces[0].top}, right=${faces[0].right}, bottom=${faces[0].bottom}")
                    processFace(faces[0], imageProxy)
                } else {
                    Log.d(TAG, "No faces detected in frame $frameCount")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error in frame $frameCount analysis", e)
            } finally {
                imageProxy.close()
            }
        }
    }
    
    // Very basic detection that should find any large rectangular area
    private fun detectFacesBasic(bitmap: Bitmap): List<RectF> {
        try {
            Log.d(TAG, "Starting BASIC detection (analyzing actual pixels)")
            
            // Try to detect actual contrast in the image
            val faces = mutableListOf<RectF>()
            val width = bitmap.width
            val height = bitmap.height
            
            // Sample the bitmap to find regions with contrast
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            
            // Calculate average brightness
            var totalBrightness = 0L
            for (pixel in pixels) {
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                totalBrightness += (r + g + b) / 3
            }
            val avgBrightness = totalBrightness / pixels.size
            
            Log.d(TAG, "Image average brightness: $avgBrightness")
            
            // Look for regions that are significantly different from average
            val gridSize = 40 // Check 40x40 pixel regions
            val threshold = 30 // Brightness difference threshold
            
            for (y in 0 until height - gridSize step gridSize/2) {
                for (x in 0 until width - gridSize step gridSize/2) {
                    var regionBrightness = 0L
                    var pixelCount = 0
                    
                    // Calculate average brightness for this region
                    for (ry in y until minOf(y + gridSize, height)) {
                        for (rx in x until minOf(x + gridSize, width)) {
                            val pixel = pixels[ry * width + rx]
                            val r = (pixel shr 16) and 0xFF
                            val g = (pixel shr 8) and 0xFF
                            val b = pixel and 0xFF
                            regionBrightness += (r + g + b) / 3
                            pixelCount++
                        }
                    }
                    
                    if (pixelCount > 0) {
                        val regionAvg = regionBrightness / pixelCount
                        val contrast = kotlin.math.abs(regionAvg - avgBrightness)
                        
                        // If this region has significant contrast, consider it a potential face
                        if (contrast > threshold) {
                            val face = RectF(
                                x.toFloat(),
                                y.toFloat(),
                                (x + gridSize).toFloat(),
                                (y + gridSize).toFloat()
                            )
                            faces.add(face)
                            Log.d(TAG, "Found contrasted region at $x,$y with contrast $contrast")
                        }
                    }
                }
            }
            
            // If no contrast regions found, return empty (let other methods try)
            if (faces.isEmpty()) {
                Log.d(TAG, "No contrasted regions found in basic detection")
                return emptyList()
            }
            
            // Sort by size and return largest regions
            val sortedFaces = faces.sortedByDescending { 
                (it.right - it.left) * (it.bottom - it.top) 
            }.take(3)
            
            Log.d(TAG, "Basic detection found ${sortedFaces.size} contrasted regions")
            return sortedFaces
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in basic detection", e)
            return emptyList()
        }
    }

    private fun processFace(faceRect: RectF, imageProxy: ImageProxy) {
        // Get eye regions based on face geometry
        val eyeRegions = contrastFaceDetector.getEyeRegions(faceRect)
        
        if (eyeRegions == null) {
            Log.d(TAG, "Could not get eye regions from face")
            return
        }
        
        val (leftEyeRect, rightEyeRect) = eyeRegions

        val leftEyeBitmap = cropAndConvert(imageProxy, leftEyeRect)
        val rightEyeBitmap = cropAndConvert(imageProxy, rightEyeRect)
        
        if (::tflite.isInitialized) {
            val gaze = estimateGaze(leftEyeBitmap, rightEyeBitmap)
            gazeHistory.add(gaze)
            if (gazeHistory.size > GAZE_HISTORY_SIZE) gazeHistory.removeAt(0)

            val avgGaze = Pair(
                gazeHistory.map { it.first }.average().toFloat(),
                gazeHistory.map { it.second }.average().toFloat()
            )

            runOnUiThread {
                overlayView.updateGazePoint(avgGaze.first, avgGaze.second)
            }
        } else {
            Log.d(TAG, "TensorFlow model not loaded, showing center gaze point")
            runOnUiThread {
                overlayView.updateGazePoint(0.5f, 0.5f)
            }
        }
    }

    private fun cropAndConvert(imageProxy: ImageProxy, rect: RectF): Bitmap {
        val matrix = Matrix().apply {
            postRotate(-imageProxy.imageInfo.rotationDegrees.toFloat())
            postScale(1f, -1f)
        }
        
        val bitmap = imageProxy.toBitmap()
        if (bitmap == null) {
            return Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.RGB_565)
        }
        
        // Ensure crop coordinates are within bitmap bounds
        val left = rect.left.toInt().coerceAtLeast(0)
        val top = rect.top.toInt().coerceAtLeast(0)
        val right = rect.right.toInt().coerceAtMost(bitmap.width)
        val bottom = rect.bottom.toInt().coerceAtMost(bitmap.height)
        val width = (right - left).coerceAtLeast(1)
        val height = (bottom - top).coerceAtLeast(1)
        
        return try {
            Bitmap.createScaledBitmap(
                Bitmap.createBitmap(bitmap, left, top, width, height, matrix, true),
                INPUT_SIZE,
                INPUT_SIZE,
                true
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error cropping bitmap", e)
            // Return a default bitmap if cropping fails
            Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.RGB_565)
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        return try {
            val buffer = planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            Log.d(TAG, "Converted ImageProxy to bitmap: ${bitmap?.width}x${bitmap?.height}")
            bitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to bitmap", e)
            null
        }
    }

    private fun estimateGaze(leftEye: Bitmap, rightEye: Bitmap): Pair<Float, Float> {
        val inputBuffer = ByteBuffer.allocateDirect(2 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())
        
        addEyeToBuffer(leftEye, inputBuffer)
        addEyeToBuffer(rightEye, inputBuffer)
        
        val output = Array(1) { FloatArray(2) }
        tflite.run(inputBuffer, output)
        return Pair(output[0][0], output[0][1])
    }

    private fun addEyeToBuffer(bitmap: Bitmap, buffer: ByteBuffer) {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16 and 0xFF) - 127.5f) / 127.5f) // R
            buffer.putFloat(((pixel shr 8 and 0xFF) - 127.5f) / 127.5f)  // G
            buffer.putFloat(((pixel and 0xFF) - 127.5f) / 127.5f)         // B
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (allPermissionsGranted()) {
                Log.d(TAG, "Camera permission granted, starting camera")
                if (isOpenCVInitialized) {
                    startCamera()
                } else {
                    Log.d(TAG, "Waiting for OpenCV initialization to complete")
                }
            } else {
                Log.e(TAG, "Camera permission not granted")
                showTestPattern()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::tflite.isInitialized) {
            tflite.close()
        }
        if (::contrastFaceDetector.isInitialized) {
            contrastFaceDetector.release()
        }
        cameraExecutor.shutdown()
    }
}