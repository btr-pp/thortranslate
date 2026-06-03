package com.kanjilens.analysis

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/**
 * Offline English dictionary backed by a trimmed ECDICT asset (`ecdict.json`).
 *
 * Each asset entry is `[word, phonetic, translation(zh), definition(en), pos, tag]`.
 * The asset is loaded lazily on first lookup so it does not compete for memory
 * with the Japanese JMDict (`dictionary.json`) when only JP mode is used.
 *
 * Build the asset with `tools/build_ecdict.py`.
 */
class EnglishDictionaryLookup(private val context: Context) {

    companion object {
        private const val TAG = "KanjiLens"
        private const val ASSET = "ecdict.json"
    }

    data class EcdictEntry(
        val word: String,
        val phonetic: String,
        val translation: String,
        val definition: String,
        val pos: String,
        val tag: String,
    )

    private val byWord: Map<String, EcdictEntry> by lazy { load() }

    private fun load(): Map<String, EcdictEntry> {
        Log.d(TAG, "EnglishDictionary: Loading...")
        val json = context.assets.open(ASSET).bufferedReader().readText()
        val type = object : TypeToken<List<List<String>>>() {}.type
        val entries: List<List<String>> = Gson().fromJson(json, type)

        val map = HashMap<String, EcdictEntry>(entries.size)
        for (entry in entries) {
            if (entry.size < 6) continue
            val word = entry[0]
            map.putIfAbsent(
                word,
                EcdictEntry(
                    word = word,
                    phonetic = entry[1],
                    translation = entry[2],
                    definition = entry[3],
                    pos = entry[4],
                    tag = entry[5],
                ),
            )
        }
        Log.d(TAG, "EnglishDictionary: Loaded ${map.size} entries")
        return map
    }

    fun lookup(word: String): EcdictEntry? = byWord[word.lowercase()]
}
