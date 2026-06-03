package com.kanjilens.analysis

/**
 * Splits recognized English (Latin-script) text into word tokens.
 *
 * Unlike Japanese, English is whitespace-delimited, so no Kuromoji is needed.
 * We split on whitespace, then trim surrounding punctuation while preserving the
 * original casing for display. The dictionary lookup itself lowercases the word.
 */
class EnglishTokenizer {

    fun tokenize(text: String): List<String> {
        return text
            .split(Regex("\\s+"))
            .map { it.trim(*TRIM_CHARS) }
            .filter { it.isNotEmpty() && it.any(Char::isLetter) }
    }

    companion object {
        private val TRIM_CHARS = charArrayOf(
            '.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']',
            '{', '}', '<', '>', '/', '\\', '*', '_', '~', '`', '|',
            '“', '”', '‘', '’', // smart quotes
            '—', '–', // em/en dash
        )
    }
}
