package com.example.sibi

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import com.example.sibi.ml.MetadataSibi
import org.tensorflow.lite.support.image.TensorImage

class ObjectDetectionHelper(
    val context: Context
) {
    private var model: MetadataSibi? = null

    init {
        model = MetadataSibi.newInstance(context)
    }

    // Function to run object detection on a given bitmap
    fun detectObjects(bitmap: Bitmap): List<DetectionResult> {
        val tensorImage = TensorImage.fromBitmap(bitmap)

        // Run the model inference
        val outputs = model?.process(tensorImage)

        // Get the first detection result (you can modify this to process multiple detections)
        val detectionResultList = outputs?.detectionResultList

        val detectionResults = mutableListOf<DetectionResult>()
        detectionResultList?.forEach { detectionResult ->
            // Extract the detection data
            val score = detectionResult.scoreAsFloat
            val location = detectionResult.locationAsRectF
            val category = detectionResult.categoryAsString

            // Store the result
            detectionResults.add(
                DetectionResult(score = score, location = location, category = category)
            )
        }

        return detectionResults
    }

    fun close() {
        model?.close()  // Free up resources when done with the model
    }

    data class DetectionResult(
        val score: Float,
        val location: RectF,
        val category: String
    )
}
