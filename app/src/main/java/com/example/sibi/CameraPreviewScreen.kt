package com.example.sibi

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.sibi.ml.MetadataSibi
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.model.Model
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import android.Manifest

@Composable
fun CameraPreviewScreen(modifier: Modifier = Modifier) {
    val lensFacing = CameraSelector.LENS_FACING_BACK
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current
    var showPermissionDialog by remember { mutableStateOf(false) }
    val preview = Preview.Builder().build()
    val previewView = remember { PreviewView(context) }
    val boundingBoxOverlay = remember { OverlayView(context) }
    val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    val sibiModel = remember {
        MetadataSibi.newInstance(
            context,
            Model.Options.Builder()
                .setNumThreads(4)
                .build()
        )
    }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
        if (!isGranted) {
            // Tampilkan dialog untuk pengguna jika izin tidak diberikan
            showPermissionDialog = true
        }
    }

    // Request camera permission if not granted
    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (hasCameraPermission) {
        LaunchedEffect(lensFacing) {
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(analysisExecutor) { imageProxy ->
                            imageProxy.use { proxy ->
                                processImageProxy(proxy, context, boundingBoxOverlay, previewView, sibiModel)
                            }
                        }
                    }

                cameraProvider.bindToLifecycle(
                    lifecycleOwner, cameraSelector, preview, imageAnalyzer
                )
            }, ContextCompat.getMainExecutor(context))

            preview.setSurfaceProvider(previewView.surfaceProvider)
        }

        DisposableEffect(Unit) {
            onDispose {
                analysisExecutor.shutdown()
                sibiModel.close()
            }
        }

        Box(modifier = modifier.fillMaxSize()) {
            AndroidView({ previewView }, modifier = Modifier.fillMaxSize())
            AndroidView({ boundingBoxOverlay }, modifier = Modifier.fillMaxSize())
        }
    } else {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = androidx.compose.ui.Alignment.Center
        ) {
            Text("Izin kamera diperlukan untuk menggunakan fitur ini.")
        }
    }
}

private fun processImageProxy(
    imageProxy: ImageProxy,
    context: Context,
    overlayView: OverlayView,
    previewView: PreviewView,
    model: MetadataSibi
) {
    val bitmap = imageProxy.toBitmap()

    val previewAspectRatio = previewView.width.toFloat() / previewView.height
    val matrix = Matrix().apply { postRotate(90f) }
    val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    val scaledBitmap = Bitmap.createScaledBitmap(
        rotatedBitmap,
        (rotatedBitmap.height * previewAspectRatio).toInt(),
        rotatedBitmap.height,
        true
    )

    val image = TensorImage.fromBitmap(scaledBitmap)
    val outputs = model.process(image)
    val detectionResult = outputs.detectionResultList

    overlayView.post {
        overlayView.updateResults(detectionResult, scaledBitmap.width, scaledBitmap.height)
    }
}

fun ImageProxy.toBitmap(): Bitmap {
    val planes = this.planes
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer[nv21, 0, ySize]
    vBuffer[nv21, ySize, vSize]
    uBuffer[nv21, ySize + vSize, uSize]
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

