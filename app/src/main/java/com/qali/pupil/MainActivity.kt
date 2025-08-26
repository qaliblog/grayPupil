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
        
        previewView = findViewById(R.id.previewView)
        overlayView = GazeOverlayView(this)
        findViewById<android.widget.FrameLayout>(R.id.overlayContainer).addView(overlayView)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Initialize OpenCV first, then start camera
        initializeOpenCV()

        if (allPermissionsGranted()) {
            // Camera will be started after OpenCV initialization
        } else {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        }
    }
    
    private fun initializeOpenCV() {
        com.qali.pupil.OpenCVLoader.initializeOpenCV(this) {
            runOnUiThread {
                isOpenCVInitialized = true
                contrastFaceDetector = ContrastFaceDetector()
                
                try {
                    tflite = Interpreter(loadModelFile("gaze_model.tflite"))
                    if (allPermissionsGranted()) {
                        startCamera()
                    }
                } catch (e: IOException) {
                    Log.e(TAG, "Error loading TensorFlow Lite model", e)
                }
            }
        }
    }

    private fun allPermissionsGranted() = 
        ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        if (!isOpenCVInitialized) {
            Log.w(TAG, "OpenCV not initialized yet, waiting...")
            return
        }
        
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
                    it.setAnalyzer(cameraExecutor, ContrastFaceAnalyzer())
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

    private inner class ContrastFaceAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (!isOpenCVInitialized) {
                imageProxy.close()
                return
            }
            
            try {
                val bitmap = imageProxy.toBitmap()
                val faces = contrastFaceDetector.detectFaces(bitmap)
                
                if (faces.isNotEmpty()) {
                    processFace(faces[0], imageProxy)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in face analysis", e)
            } finally {
                imageProxy.close()
            }
        }
    }

    private fun processFace(faceRect: RectF, imageProxy: ImageProxy) {
        // Get eye regions based on face geometry
        val eyeRegions = contrastFaceDetector.getEyeRegions(faceRect)
        
        if (eyeRegions == null) return
        
        val (leftEyeRect, rightEyeRect) = eyeRegions

        val leftEyeBitmap = cropAndConvert(imageProxy, leftEyeRect)
        val rightEyeBitmap = cropAndConvert(imageProxy, rightEyeRect)
        
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
    }

    private fun cropAndConvert(imageProxy: ImageProxy, rect: RectF): Bitmap {
        val matrix = Matrix().apply {
            postRotate(-imageProxy.imageInfo.rotationDegrees.toFloat())
            postScale(1f, -1f)
        }
        
        val bitmap = imageProxy.toBitmap()
        
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

    private fun ImageProxy.toBitmap(): Bitmap {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
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
            if (allPermissionsGranted() && isOpenCVInitialized) {
                startCamera()
            } else {
                Log.e(TAG, "Camera permission not granted")
                finish()
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