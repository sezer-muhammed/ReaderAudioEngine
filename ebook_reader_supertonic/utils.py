import re
import numpy as np

AVAILABLE_LANGS = ["en", "ko", "es", "pt", "fr"]

def preprocess_text(text, lang=None):
    # Normalize unicode characters
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    
    # Remove emojis (common patterns)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\U0001F1E6-\U0001F1FF]+', '', text)
    
    # Replace various dashes and symbols
    replacements = {
        "–": "-",
        "‑": "-",
        "—": "-",
        "_": " ",
        "\u201C": '"',
        "\u201D": '"',
        "\u2018": "'",
        "\u2019": "'",
        "´": "'",
        "`": "'",
        "[": " ",
        "]": " ",
        "|": " ",
        "/": " ",
        "#": " ",
        "→": " ",
        "←": " ",
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove special symbols
    # Remove common decorative / copyright symbols
    text = re.sub(r'[♥♡❤☆★©®™✓✔✕✖]', "", text)


    # Replace known expressions
    expr_replacements = {
        # Symbols
        "@": " at ",
        "&": " and ",
        "%": " percent ",
        "#": " number ",
        "$": " dollar ",

        # Latin abbreviations
        "e.g.": "for example",
        "e.g.,": "for example",
        "i.e.": "that is",
        "i.e.,": "that is",
        "etc.": "and so on",
        "vs.": "versus",

        # Academic / formal
        "cf.": "compare",
        "et al.": "and others",

        # Informal / contractions
        "w/": "with",
        "w/o": "without",
    }

    
    for k, v in expr_replacements.items():
        text = text.replace(k, v)
    
    # Fix spacing around punctuation
    text = re.sub(r' ,', ",", text)
    text = re.sub(r' \.', ".", text)
    text = re.sub(r' !', "!", text)
    text = re.sub(r' \?', "?", text)
    text = re.sub(r' ;', ";", text)
    text = re.sub(r' :', ":", text)
    text = re.sub(r' \'', "'", text)
    
    # Remove duplicate quotes
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    while "``" in text:
        text = text.replace("``", "`")
    
    # Remove extra spaces
    text = re.sub(r'\s+', " ", text).strip()

    # If text doesn't end with punctuation/etc, add a period
    if not re.search(r'[.!?;:,\'"\)\]}…。」』】〉》›»]$', text):
        text += "."
    
    # Add language tags
    processed_text = text
    if lang is not None:
        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        processed_text = f"<{lang}>{text}</{lang}>"
    else:
        processed_text = f"<na>{text}</na>"
    
    return processed_text, text

class UnicodeProcessor:
    def __init__(self, indexer):
        self.indexer = indexer

    def __call__(self, text_list, lang=None):
        results = [preprocess_text(t, lang) for t in text_list]
        processed_texts = [r[0] for r in results]
        original_texts = [r[1] for r in results]
        
        text_ids_lengths = [len(t) for t in processed_texts]
        max_len = max(text_ids_lengths)
        
        text_ids = []
        unsupported_chars = set()
        
        for i in range(len(processed_texts)):
            row = [0] * max_len
            unicode_vals = [ord(char) for char in processed_texts[i]]
            for j in range(len(unicode_vals)):
                char_code = unicode_vals[j]
                
                index_value = -1
                if char_code < len(self.indexer):
                    index_value = self.indexer[char_code]
                
                if index_value == -1:
                    unsupported_chars.add(processed_texts[i][j])
                    row[j] = 0
                else:
                    row[j] = index_value
            text_ids.append(row)
        
        text_mask = self.get_text_mask(text_ids_lengths, max_len)
        return {
            'text_ids': np.array(text_ids, dtype=np.int64),
            'text_mask': text_mask,
            'unsupported_chars': list(unsupported_chars),
            'processed_texts': processed_texts,
            'original_texts': original_texts
        }

    def get_text_mask(self, lengths, max_len=None):
        if max_len is None:
            max_len = max(lengths)
        mask = np.zeros((len(lengths), 1, max_len), dtype=np.float32)
        for i, length in enumerate(lengths):
            mask[i, 0, :length] = 1.0
        return mask

def detect_language(text):
    if not text or len(text.strip()) < 3:
        return None
    
    sample_text = text[-100:] if len(text) > 100 else text
    import unicodedata
    normalized_text = unicodedata.normalize('NFC', sample_text).lower()
    
    # Korean detection
    korean_regex = r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]'
    korean_matches = re.findall(korean_regex, normalized_text)
    if len(korean_matches) >= 2:
        return 'ko'
    
    scores = { 'en': 0, 'es': 0, 'fr': 0, 'pt': 0 }
    
    # Distinctive chars
    if 'ñ' in normalized_text: scores['es'] += 15
    if re.search(r'[¿¡]', normalized_text): scores['es'] += 12
    if 'ã' in normalized_text: scores['pt'] += 15
    if 'õ' in normalized_text: scores['pt'] += 15
    if 'œ' in normalized_text: scores['fr'] += 15
    if re.search(r'[ùû]', normalized_text): scores['fr'] += 10
    
    if 'ç' in normalized_text:
        scores['fr'] += 4
        scores['pt'] += 4
    
    # Stopwords (simplified)
    exclusive_words = {
        'en': ['the', 'is', 'are', 'was', 'were', 'have'],
        'es': ['el', 'los', 'las', 'es', 'está'],
        'fr': ['le', 'les', 'est', 'sont', 'dans'],
        'pt': ['os', 'as', 'é', 'são', 'está']
    }
    
    words = re.findall(r'[a-zàâãäåçéèêëíìîïñóòôõöúùûüýÿœæ]+', normalized_text)
    for word in words:
        for lang, word_list in exclusive_words.items():
            if word in word_list:
                scores[lang] += 3
                
    detected_lang = max(scores, key=scores.get)
    if scores[detected_lang] >= 4:
        return detected_lang
    return None
