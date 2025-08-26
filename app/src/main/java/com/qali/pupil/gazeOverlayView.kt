package com.qali.pupil

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import org.opencv.core.MatOfPoint
import org.opencv.core.Point

class GazeOverlayView(context: Context) : View(context) {
    private var gazeX: Float = 0f
    private var gazeY: Float = 0f
    private var contours: List<MatOfPoint> = emptyList()
    
    private val gazePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val contourPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    fun updateGazePoint(x: Float, y: Float, detectedContours: List<MatOfPoint> = emptyList()) {
        gazeX = x * width
        gazeY = y * height
        contours = detectedContours
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
        
        // Draw contours as green squares
        for (contour in contours) {
            val points = contour.toArray()
            if (points.isNotEmpty()) {
                // For each contour, draw green squares at key points
                val step = maxOf(1, points.size / 10) // Sample up to 10 points per contour
                for (i in points.indices step step) {
                    val point = points[i]
                    val x = (point.x * width / 100).toFloat() // Scale appropriately
                    val y = (point.y * height / 100).toFloat() // Scale appropriately
                    
                    val smallSquareSize = 15f
                    canvas.drawRect(
                        x - smallSquareSize / 2,
                        y - smallSquareSize / 2,
                        x + smallSquareSize / 2,
                        y + smallSquareSize / 2,
                        contourPaint
                    )
                }
            }
        }
    }
}