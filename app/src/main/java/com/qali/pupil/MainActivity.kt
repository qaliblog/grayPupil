package com.qali.pupil


import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
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
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Size as OpenCVSize
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private var tflite: Interpreter? = null
    private lateinit var overlayView: GazeOverlayView
    private var isOpenCVLoaded = false
    private var faceContours = mutableListOf<MatOfPoint>()

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
    
    // Face detection components
    private val detector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        FaceDetection.getClient(options)
    }

    // Model parameters
    private val INPUT_SIZE = 64
    private val GAZE_HISTORY_SIZE = 5
    private val gazeHistory = mutableListOf<Pair<Float, Float>>()

    companion object {
        private const val TAG = "EyeTracking"
        private const val CAMERA_PERMISSION_CODE = 101
        private const val LEFT_EYE = 2  // FaceLandmark.LEFT_EYE
        private const val RIGHT_EYE = 7 // FaceLandmark.RIGHT_EYE
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        previewView = findViewById(R.id.previewView)
        overlayView = GazeOverlayView(this)
        findViewById<android.widget.FrameLayout>(R.id.overlayContainer).addView(overlayView)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Try to load gaze model, but don't crash if it's missing
        try {
            tflite = Interpreter(loadModelFile("gaze_model.tflite"))
            Log.d(TAG, "Gaze model loaded successfully")
        } catch (e: Exception) {
            Log.w(TAG, "Gaze model not found, using mock gaze estimation: ${e.message}")
            tflite = null
        }

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
                    it.setAnalyzer(cameraExecutor, FaceAnalyzer())
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

    private inner class FaceAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            val mediaImage = imageProxy.image ?: return
            val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
            
            detector.process(image)
                .addOnSuccessListener { faces: List<Face> ->
                    if (faces.isNotEmpty()) processFace(faces[0], imageProxy)
                }
                .addOnCompleteListener { imageProxy.close() }
        }
    }

    private fun processFace(face: Face, imageProxy: ImageProxy) {
        val leftEye = face.getLandmark(LEFT_EYE)?.position
        val rightEye = face.getLandmark(RIGHT_EYE)?.position
        
        if (leftEye == null || rightEye == null) return

        val eyeSize = (face.boundingBox.width() * 0.18).toInt()
        val leftEyeRect = RectF(
            leftEye.x - eyeSize,
            leftEye.y - eyeSize,
            leftEye.x + eyeSize,
            leftEye.y + eyeSize
        )
        val rightEyeRect = RectF(
            rightEye.x - eyeSize,
            rightEye.y - eyeSize,
            rightEye.x + eyeSize,
            rightEye.y + eyeSize
        )

        val leftEyeBitmap = cropAndConvert(imageProxy, leftEyeRect)
        val rightEyeBitmap = cropAndConvert(imageProxy, rightEyeRect)
        
        // Get the full face area for contour detection
        val faceBitmap = cropAndConvert(imageProxy, face.boundingBox)
        
        // Apply contrast enhancement and detect face contours
        val enhancedFace = enhanceContrast(faceBitmap)
        val detectedContours = detectFaceContours(enhancedFace)
        
        // Store contours for visualization
        faceContours.clear()
        faceContours.addAll(detectedContours)
        
        val gaze = estimateGaze(leftEyeBitmap, rightEyeBitmap)
        gazeHistory.add(gaze)
        if (gazeHistory.size > GAZE_HISTORY_SIZE) gazeHistory.removeAt(0)

        val avgGaze = Pair(
            gazeHistory.map { it.first }.average().toFloat(),
            gazeHistory.map { it.second }.average().toFloat()
        )

        runOnUiThread {
            overlayView.updateGazeAndContours(avgGaze.first, avgGaze.second, faceContours, face.boundingBox)
        }
    }

    private fun cropAndConvert(imageProxy: ImageProxy, rect: Rect): Bitmap {
        val matrix = Matrix().apply {
            postRotate(-imageProxy.imageInfo.rotationDegrees.toFloat())
            postScale(1f, -1f)
        }
        
        val bitmap = imageProxy.toBitmap()
        return Bitmap.createScaledBitmap(
            Bitmap.createBitmap(bitmap, 
                rect.left, 
                rect.top,
                rect.width(),
                rect.height(),
                matrix,
                true),
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )
    }

    private fun cropAndConvert(imageProxy: ImageProxy, rect: RectF): Bitmap {
        val matrix = Matrix().apply {
            postRotate(-imageProxy.imageInfo.rotationDegrees.toFloat())
            postScale(1f, -1f)
        }
        
        val bitmap = imageProxy.toBitmap()
        return Bitmap.createScaledBitmap(
            Bitmap.createBitmap(bitmap, 
                rect.left.toInt(), 
                rect.top.toInt(),
                rect.width().toInt(),
                rect.height().toInt(),
                matrix,
                true),
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun estimateGaze(leftEye: Bitmap, rightEye: Bitmap): Pair<Float, Float> {
        return tflite?.let { interpreter ->
            val inputBuffer = ByteBuffer.allocateDirect(2 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
                .order(ByteOrder.nativeOrder())
            
            addEyeToBuffer(leftEye, inputBuffer)
            addEyeToBuffer(rightEye, inputBuffer)
            
            val output = Array(1) { FloatArray(2) }
            interpreter.run(inputBuffer, output)
            Pair(output[0][0], output[0][1])
        } ?: run {
            // Mock gaze estimation - return center of screen
            Pair(0.5f, 0.5f)
        }
    }

    private fun addEyeToBuffer(bitmap: Bitmap, buffer: ByteBuffer) {
        if (tflite == null) return // Skip if no model available
        
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16 and 0xFF) - 127.5f) / 127.5f) // R
            buffer.putFloat(((pixel shr 8 and 0xFF) - 127.5f) / 127.5f)  // G
            buffer.putFloat(((pixel and 0xFF) - 127.5f) / 127.5f)         // B
        }
    }

    private fun enhanceContrast(bitmap: Bitmap): Bitmap {
        if (!isOpenCVLoaded) return bitmap
        
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

    private fun detectFaceContours(bitmap: Bitmap): List<MatOfPoint> {
        if (!isOpenCVLoaded) return emptyList()
        
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
            area > 500.0 // Minimum area for face features
        }
        
        mat.release()
        grayMat.release()
        enhancedMat.release()
        blurredMat.release()
        edgesMat.release()
        hierarchy.release()
        
        return filteredContours
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

    override fun onDestroy() {
        super.onDestroy()
        tflite?.close()
        cameraExecutor.shutdown()
    }
}