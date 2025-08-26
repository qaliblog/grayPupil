package com.qali.pupil

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import android.view.View
// OpenCV imports temporarily removed

class GazeOverlayView(context: Context) : View(context) {
    private var gazeX: Float = 0f
    private var gazeY: Float = 0f
    // OpenCV contour functionality temporarily disabled
    
    private val gazePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // contourPaint temporarily removed

    fun updateGazePoint(x: Float, y: Float) {
        if (width > 0 && height > 0) {
            gazeX = x * width
            gazeY = y * height
            Log.d("GazeOverlay", "updateGazePoint: input($x, $y) -> screen($gazeX, $gazeY), view size: ${width}x${height}")
            invalidate()
        } else {
            Log.w("GazeOverlay", "View not ready: ${width}x${height}, deferring update")
        }
    }
    
    // updateGazeAndContours temporarily disabled - OpenCV functionality removed

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        Log.d("GazeOverlay", "onDraw: drawing at ($gazeX, $gazeY) on canvas ${canvas.width}x${canvas.height}")
        
        // Draw green square at gaze point
        val squareSize = 30f
        canvas.drawRect(
            gazeX - squareSize / 2,
            gazeY - squareSize / 2,
            gazeX + squareSize / 2,
            gazeY + squareSize / 2,
            gazePaint
        )
        
        // Contour drawing temporarily disabled - will re-add with OpenCV
    }
}