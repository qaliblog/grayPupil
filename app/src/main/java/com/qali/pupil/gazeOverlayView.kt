package com.qali.pupil

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.view.View
import org.opencv.core.MatOfPoint
import org.opencv.core.Point

class GazeOverlayView(context: Context) : View(context) {
    private var gazeX: Float = 0f
    private var gazeY: Float = 0f
    private var contours: List<MatOfPoint> = emptyList()
    private var faceRect: Rect? = null
    
    private val gazePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val contourPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    fun updateGazePoint(x: Float, y: Float) {
        gazeX = x * width
        gazeY = y * height
        invalidate()
    }
    
    fun updateGazeAndContours(x: Float, y: Float, detectedContours: List<MatOfPoint>, faceBounds: Rect) {
        gazeX = x * width
        gazeY = y * height
        contours = detectedContours
        faceRect = faceBounds
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
        
        // Draw green squares for face contours
        faceRect?.let { face ->
            for (contour in contours) {
                val points = contour.toArray()
                if (points.isNotEmpty()) {
                    // Draw green squares at contour points, scaled to face position
                    val step = maxOf(1, points.size / 15) // Sample up to 15 points per contour
                    for (i in points.indices step step) {
                        val point = points[i]
                        
                        // Scale contour coordinates to face position on screen
                        val scaleX = width.toFloat() / 640f // Assuming camera resolution
                        val scaleY = height.toFloat() / 480f
                        
                        val x = (face.left + point.x * face.width() / 64) * scaleX // Scale from INPUT_SIZE
                        val y = (face.top + point.y * face.height() / 64) * scaleY
                        
                        val contourSquareSize = 8f
                        canvas.drawRect(
                            x - contourSquareSize / 2,
                            y - contourSquareSize / 2,
                            x + contourSquareSize / 2,
                            y + contourSquareSize / 2,
                            contourPaint
                        )
                    }
                }
            }
        }
    }
}