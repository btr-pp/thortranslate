package com.kanjilens.capture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.HandlerThread
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

class ScreenCaptureManager(private val context: Context) {

    companion object {
        private const val TAG = "KanjiLens"
        private const val CAPTURE_TIMEOUT_MS = 2000L
    }

    private var mediaProjection: MediaProjection? = null

    // Capture resources are created once and REUSED across cycles. Recreating a
    // VirtualDisplay every second (as auto-translate does) churns a scarce system
    // resource and, combined with screen lock/unlock power transitions, can wedge
    // the display subsystem and freeze the whole device.
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var captureWidth = 0
    private var captureHeight = 0
    private var captureDensity = 0

    // Run ImageReader callbacks and the full-screen pixel copy off the main thread.
    private val captureThread = HandlerThread("ScreenCapture").apply { start() }
    private val handler = Handler(captureThread.looper)

    private var projectionCallback: MediaProjection.Callback? = null

    // Callback for when projection is ready
    private var onProjectionReady: (() -> Unit)? = null

    val projectionManager: MediaProjectionManager =
        context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

    val isReady: Boolean
        get() = mediaProjection != null

    fun setProjection(projection: MediaProjection) {
        Log.d(TAG, "MediaProjection received")
        mediaProjection = projection
        // The system can revoke the projection (e.g. another app starts capturing).
        // React to that instead of silently using a dead projection.
        val callback = object : MediaProjection.Callback() {
            override fun onStop() {
                Log.d(TAG, "MediaProjection stopped by system")
                release()
            }
        }
        projection.registerCallback(callback, handler)
        projectionCallback = callback
        onProjectionReady?.invoke()
        onProjectionReady = null
    }

    fun awaitProjectionReady(callback: () -> Unit) {
        if (mediaProjection != null) {
            callback()
        } else {
            onProjectionReady = callback
        }
    }

    /**
     * Capture the current screen.
     *
     * The persistent VirtualDisplay is kept with NO output surface between captures, so
     * it does no mirroring/compositing while idle. We attach the ImageReader surface only
     * for the brief moment needed to grab one frame, then detach again. This avoids both
     * per-second VirtualDisplay churn AND continuous full-screen mirroring — the latter
     * matters a lot under a running game, where the compositor (SurfaceFlinger) is already
     * saturated and an always-on mirror can tip the whole device into a freeze.
     *
     * @param forceFreshFrame retained for source compatibility; capture always renders a
     *   fresh frame now (attaching the surface re-composites current screen content).
     */
    @Suppress("UNUSED_PARAMETER")
    suspend fun captureScreen(forceFreshFrame: Boolean = true): Bitmap? {
        val projection = mediaProjection
        if (projection == null) {
            Log.e(TAG, "captureScreen called but mediaProjection is null")
            return null
        }

        val metrics = getScreenMetrics()
        if (!ensureCaptureResources(projection, metrics)) return null
        val reader = imageReader ?: return null
        val display = virtualDisplay ?: return null

        val width = captureWidth
        val height = captureHeight

        // All surface toggles and ImageReader reads run on the single capture handler
        // thread (FIFO ordered) so attach/detach never race the frame reads.
        return withTimeoutOrNull(CAPTURE_TIMEOUT_MS) {
            try {
                suspendCancellableCoroutine { continuation ->
                    val tryAcquire = Runnable {
                        if (continuation.isActive) {
                            val image = reader.acquireLatestImage()
                            if (image != null) {
                                reader.setOnImageAvailableListener(null, handler)
                                val bitmap = imageToBitmap(image, width, height)
                                image.close()
                                if (continuation.isActive) continuation.resume(bitmap)
                            }
                        }
                    }
                    reader.setOnImageAvailableListener({ tryAcquire.run() }, handler)
                    // Attach the output surface to render the current screen into our
                    // reader; detached again in the finally below once we have a frame.
                    handler.post { runCatching { display.setSurface(reader.surface) } }
                    continuation.invokeOnCancellation {
                        reader.setOnImageAvailableListener(null, handler)
                    }
                }
            } finally {
                // Stop mirroring as soon as we're done so we don't keep loading the GPU,
                // then drain any frames the source pushed before the detach took effect —
                // otherwise a full buffer would stall the producer and the next capture
                // (its listener never fires) would time out.
                handler.post {
                    runCatching {
                        display.setSurface(null)
                        var leftover = reader.acquireLatestImage()
                        while (leftover != null) {
                            leftover.close()
                            leftover = reader.acquireLatestImage()
                        }
                    }
                }
            }
        }
    }

    /** Ensure an ImageReader + VirtualDisplay exist for the given screen size. */
    private fun ensureCaptureResources(
        projection: MediaProjection,
        metrics: DisplayMetrics,
    ): Boolean {
        val width = metrics.widthPixels
        val height = metrics.heightPixels
        val density = metrics.densityDpi

        if (imageReader != null && virtualDisplay != null &&
            width == captureWidth && height == captureHeight && density == captureDensity
        ) {
            return true
        }

        // Size/orientation changed or first use — rebuild from scratch.
        releaseCaptureResources()
        captureWidth = width
        captureHeight = height
        captureDensity = density

        val reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        imageReader = reader
        return try {
            val display = projection.createVirtualDisplay(
                "KanjiLensCapture",
                width, height, density,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                reader.surface,
                null, null,
            )
            // Start idle: no output surface means no mirroring until a capture attaches one.
            display?.setSurface(null)
            virtualDisplay = display
            Log.d(TAG, "Capture resources created: ${width}x${height} @ ${density}dpi")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create VirtualDisplay", e)
            releaseCaptureResources()
            false
        }
    }

    /**
     * Release the VirtualDisplay + ImageReader but keep the MediaProjection alive.
     * Call this when the screen locks / app is backgrounded so we never hold a
     * VirtualDisplay across the screen-power transition. Resources are rebuilt on the
     * next capture.
     */
    fun pauseCapture() {
        releaseCaptureResources()
    }

    private fun releaseCaptureResources() {
        virtualDisplay?.release()
        virtualDisplay = null
        imageReader?.setOnImageAvailableListener(null, handler)
        imageReader?.close()
        imageReader = null
        captureWidth = 0
        captureHeight = 0
        captureDensity = 0
    }

    fun release() {
        releaseCaptureResources()
        projectionCallback?.let { mediaProjection?.unregisterCallback(it) }
        projectionCallback = null
        mediaProjection?.stop()
        mediaProjection = null
        captureThread.quitSafely()
    }

    private fun imageToBitmap(image: android.media.Image, width: Int, height: Int): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * width

        val bitmapWidth = width + rowPadding / pixelStride
        val bitmap = Bitmap.createBitmap(bitmapWidth, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)

        return if (bitmapWidth != width) {
            val cropped = Bitmap.createBitmap(bitmap, 0, 0, width, height)
            bitmap.recycle()
            cropped
        } else {
            bitmap
        }
    }

    private fun getScreenMetrics(): DisplayMetrics {
        val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealMetrics(metrics)
        return metrics
    }
}
