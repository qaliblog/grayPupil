package com.qali.pupil

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View

class GazeOverlayView(context: Context) : View(context) {
    private var gazeX: Float = 0f
    private var gazeY: Float = 0f
    private var faceRegions: List<RectF> = emptyList()
    private var cameraSize: Pair<Int, Int>? = null
    
    private val gazePaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val facePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 40f
        isAntiAlias = true
    }

    fun updateGazePoint(x: Float, y: Float) {
        gazeX = x * width
        gazeY = y * height
        invalidate()
    }
    
    fun updateFaceRegions(regions: List<RectF>, cameraWidth: Int = 640, cameraHeight: Int = 480) {
        cameraSize = Pair(cameraWidth, cameraHeight)
        
        // Transform face regions from camera coordinates to screen coordinates
        faceRegions = regions.map { face ->
            transformCameraToScreen(face, cameraWidth, cameraHeight)
        }
        invalidate()
    }
    
    private fun transformCameraToScreen(cameraRect: RectF, cameraWidth: Int, cameraHeight: Int): RectF {
        if (width == 0 || height == 0) return cameraRect
        
        // Calculate scale factors
        val scaleX = width.toFloat() / cameraWidth.toFloat()
        val scaleY = height.toFloat() / cameraHeight.toFloat()
        
        // Simple direct mapping first (no mirroring)
        val screenRect = RectF(
            cameraRect.left * scaleX,
            cameraRect.top * scaleY,
            cameraRect.right * scaleX,
            cameraRect.bottom * scaleY
        )
        
        // Log the transformation for debugging
        android.util.Log.d("GazeOverlay", "Transform: camera(${cameraRect.left},${cameraRect.top},${cameraRect.right},${cameraRect.bottom}) -> screen(${screenRect.left},${screenRect.top},${screenRect.right},${screenRect.bottom})")
        android.util.Log.d("GazeOverlay", "Scale factors: scaleX=$scaleX, scaleY=$scaleY, screenSize=${width}x${height}, cameraSize=${cameraWidth}x${cameraHeight}")
        
        return screenRect
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Draw detected face regions
        for (i in faceRegions.indices) {
            val face = faceRegions[i]
            canvas.drawRect(face, facePaint)
            canvas.drawText("Face $i", face.left, face.top - 10, textPaint)
        }
        
        // Draw gaze point
        canvas.drawCircle(gazeX, gazeY, 20f, gazePaint)
        
        // Draw debug info
        val cameraInfo = cameraSize?.let { "(${it.first}x${it.second})" } ?: ""
        canvas.drawText("Faces: ${faceRegions.size} $cameraInfo", 50f, 100f, textPaint)
        canvas.drawText("Screen: ${width}x${height}", 50f, 150f, textPaint)
        
        // Draw coordinate system info
        if (faceRegions.isNotEmpty()) {
            val face = faceRegions[0]
            canvas.drawText("Face coords: ${face.left.toInt()},${face.top.toInt()}", 50f, 200f, textPaint)
        }
    }
}