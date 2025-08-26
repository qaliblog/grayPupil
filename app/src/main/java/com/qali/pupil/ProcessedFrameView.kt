package com.qali.pupil

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.util.AttributeSet
import android.util.Log
import android.view.View
import android.widget.ImageView
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
            
            post {
                try {
                    setImageBitmap(bitmap)
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    visibility = View.VISIBLE
                    invalidate()
                    Log.d(TAG, "Frame set to ImageView successfully")
                } catch (e: Exception) {
                    Log.e(TAG, "Error setting bitmap: ${e.message}")
                }
            }
        } else {
            Log.w(TAG, "Received null bitmap")
        }
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        Log.d(TAG, "onDraw called")
    }
}