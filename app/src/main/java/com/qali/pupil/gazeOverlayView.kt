package com.qali.pupil

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View

class GazeOverlayView(context: Context) : View(context) {
    private var gazeX: Float = 0f
    private var gazeY: Float = 0f
    
    private val gazePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    fun updateGazePoint(x: Float, y: Float) {
        gazeX = x * width
        gazeY = y * height
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Draw green square at gaze point
        val squareSize = 30f
        canvas.drawRect(
            gazeX - squareSize / 2,
            gazeY - squareSize / 2,
            gazeX + squareSize / 2,
            gazeY + squareSize / 2,
            gazePaint
        )
    }
}