package com.qali.pupil

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.util.AttributeSet
import android.util.Log
import androidx.appcompat.widget.AppCompatImageView

class ProcessedFrameView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : AppCompatImageView(context, attrs, defStyleAttr) {

    companion object {
        private const val TAG = "ProcessedFrameView"
    }

    fun updateFrame(bitmap: Bitmap?) {
        if (bitmap != null) {
            Log.d(TAG, "Updating frame: ${bitmap.width}x${bitmap.height}")
            
            // Rotate bitmap for proper orientation
            val matrix = Matrix().apply {
                postRotate(90f) // Adjust rotation as needed
            }
            
            val rotatedBitmap = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )
            
            post {
                setImageBitmap(rotatedBitmap)
                invalidate()
            }
        }
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        Log.d(TAG, "onDraw called")
    }
}