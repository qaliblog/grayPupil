package com.qali.pupil

import android.content.Context
import android.util.Log
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader

class OpenCVLoader {
    companion object {
        private const val TAG = "OpenCVLoader"
        private var isInitialized = false
        private var initializationCallback: (() -> Unit)? = null
        
        fun initializeOpenCV(context: Context, callback: () -> Unit) {
            if (isInitialized) {
                callback()
                return
            }
            
            initializationCallback = callback
            
            val loaderCallback = object : BaseLoaderCallback(context) {
                override fun onManagerConnected(status: Int) {
                    when (status) {
                        LoaderCallbackInterface.SUCCESS -> {
                            Log.d(TAG, "OpenCV loaded successfully")
                            isInitialized = true
                            initializationCallback?.invoke()
                        }
                        else -> {
                            Log.e(TAG, "OpenCV initialization failed with status: $status")
                            super.onManagerConnected(status)
                        }
                    }
                }
            }
            
            if (!OpenCVLoader.initDebug()) {
                Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, context, loaderCallback)
            } else {
                Log.d(TAG, "OpenCV library found inside package. Using it!")
                loaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
            }
        }
        
        fun isOpenCVInitialized(): Boolean = isInitialized
    }
}