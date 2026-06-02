package com.kanjilens.ocr

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer as MlKitTextRecognizer
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

class TextRecognizer {

    companion object {
        private const val TAG = "KanjiLens"

        /** OCR scripts supported by the recognizer. */
        const val SCRIPT_JAPANESE = 0
        const val SCRIPT_LATIN = 1
    }

    private val japaneseRecognizer = TextRecognition.getClient(
        JapaneseTextRecognizerOptions.Builder().build()
    )

    // Latin recognizer is only needed for English dictionary mode; create on demand.
    private var latinRecognizer: MlKitTextRecognizer? = null

    private fun recognizerFor(script: Int): MlKitTextRecognizer {
        if (script != SCRIPT_LATIN) return japaneseRecognizer
        return latinRecognizer ?: TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            .also { latinRecognizer = it }
    }

    suspend fun recognizeText(
        bitmap: Bitmap,
        script: Int = SCRIPT_JAPANESE,
    ): String? = suspendCancellableCoroutine { continuation ->
        val image = InputImage.fromBitmap(bitmap, 0)

        Log.d(TAG, "OCR: Starting text recognition on ${bitmap.width}x${bitmap.height} image (script=$script)")

        recognizerFor(script).process(image)
            .addOnSuccessListener { result ->
                val text = result.text.trim()
                Log.d(TAG, "OCR: Recognized ${result.textBlocks.size} blocks, text length=${text.length}")
                if (text.isNotEmpty()) {
                    Log.d(TAG, "OCR: Text = $text")
                    continuation.resume(text)
                } else {
                    Log.d(TAG, "OCR: No text found")
                    continuation.resume(null)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "OCR: Recognition failed", e)
                continuation.resume(null)
            }
    }

    suspend fun recognizeTextBlocks(
        bitmap: Bitmap,
        script: Int = SCRIPT_JAPANESE,
    ): List<String>? = suspendCancellableCoroutine { continuation ->
        val image = InputImage.fromBitmap(bitmap, 0)

        recognizerFor(script).process(image)
            .addOnSuccessListener { result ->
                val blocks = result.textBlocks
                    .map { it.text.trim() }
                    .filter { it.isNotEmpty() }
                if (blocks.isNotEmpty()) {
                    continuation.resume(blocks)
                } else {
                    continuation.resume(null)
                }
            }
            .addOnFailureListener {
                continuation.resume(null)
            }
    }

    fun close() {
        japaneseRecognizer.close()
        latinRecognizer?.close()
    }
}
