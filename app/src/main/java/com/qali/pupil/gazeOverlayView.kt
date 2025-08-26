package com.qali.pupil

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
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
        textSize = 30f
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
        
        Log.d("GazeOverlay", "Updated ${regions.size} face regions, transformed to ${faceRegions.size} screen regions")
        invalidate()
    }
    
    private fun transformCameraToScreen(cameraRect: RectF, cameraWidth: Int, cameraHeight: Int): RectF {
        if (width == 0 || height == 0) {
            Log.w("GazeOverlay", "Screen dimensions not available yet")
            return cameraRect
        }
        
        // Camera image aspect ratio
        val cameraAspect = cameraWidth.toFloat() / cameraHeight.toFloat()
        // Screen aspect ratio  
        val screenAspect = width.toFloat() / height.toFloat()
        
        var scaleX: Float
        var scaleY: Float
        var offsetX = 0f
        var offsetY = 0f
        
        if (cameraAspect > screenAspect) {
            // Camera is wider than screen - fit to screen height, center horizontally
            scaleY = height.toFloat() / cameraHeight.toFloat()
            scaleX = scaleY
            val scaledCameraWidth = cameraWidth * scaleX
            offsetX = (width - scaledCameraWidth) / 2f
        } else {
            // Camera is taller than screen - fit to screen width, center vertically  
            scaleX = width.toFloat() / cameraWidth.toFloat()
            scaleY = scaleX
            val scaledCameraHeight = cameraHeight * scaleY
            offsetY = (height - scaledCameraHeight) / 2f
        }
        
        // Apply transformation
        val screenRect = RectF(
            cameraRect.left * scaleX + offsetX,
            cameraRect.top * scaleY + offsetY,
            cameraRect.right * scaleX + offsetX,
            cameraRect.bottom * scaleY + offsetY
        )
        
        Log.d("GazeOverlay", "Transform camera(${cameraRect.left.toInt()},${cameraRect.top.toInt()},${cameraRect.right.toInt()},${cameraRect.bottom.toInt()}) -> screen(${screenRect.left.toInt()},${screenRect.top.toInt()},${screenRect.right.toInt()},${screenRect.bottom.toInt()})")
        Log.d("GazeOverlay", "Scales: X=$scaleX, Y=$scaleY, Offsets: X=$offsetX, Y=$offsetY")
        Log.d("GazeOverlay", "Aspects: camera=$cameraAspect, screen=$screenAspect")
        
        return screenRect
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Draw screen boundary for debugging
        val boundaryPaint = Paint().apply {
            color = Color.BLUE
            style = Paint.Style.STROKE
            strokeWidth = 2f
        }
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), boundaryPaint)
        
        // Draw detected face regions
        for (i in faceRegions.indices) {
            val face = faceRegions[i]
            canvas.drawRect(face, facePaint)
            canvas.drawText("Face $i", face.left, face.top - 5, textPaint)
        }
        
        // Draw gaze point
        canvas.drawCircle(gazeX, gazeY, 15f, gazePaint)
        
        // Draw debug info
        val cameraInfo = cameraSize?.let { "(${it.first}x${it.second})" } ?: ""
        canvas.drawText("Faces: ${faceRegions.size} $cameraInfo", 10f, 50f, textPaint)
        canvas.drawText("Screen: ${width}x${height}", 10f, 80f, textPaint)
        
        // Draw coordinate system info for first face
        if (faceRegions.isNotEmpty()) {
            val face = faceRegions[0]
            canvas.drawText("Face: (${face.left.toInt()},${face.top.toInt()})", 10f, 110f, textPaint)
        }
        
        // Draw center crosshair for reference
        val centerPaint = Paint().apply {
            color = Color.WHITE
            strokeWidth = 1f
        }
        canvas.drawLine(width/2f - 20, height/2f, width/2f + 20, height/2f, centerPaint)
        canvas.drawLine(width/2f, height/2f - 20, width/2f, height/2f + 20, centerPaint)
    }
}