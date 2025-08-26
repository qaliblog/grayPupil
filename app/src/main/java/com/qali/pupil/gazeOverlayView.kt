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
    
    private val gazePaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val facePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 3f
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
    
    fun updateFaceRegions(regions: List<RectF>) {
        faceRegions = regions
        invalidate()
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
        canvas.drawText("Faces: ${faceRegions.size}", 50f, 100f, textPaint)
    }
}