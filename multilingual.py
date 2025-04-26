import streamlit as st

# Try to import langid, but provide a fallback if it's missing
try:
    import langid
    langid_available = True
except ImportError:
    st.warning("langid package not found. Language detection will use a simplified approach.")
    langid_available = False

# Try to import transformers, but provide a fallback if it's missing
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    st.warning("transformers package not found. Advanced language detection will not be available.")
    transformers_available = False

# Initialize language identification model with caching
@st.cache_resource
def load_language_id_model():
    """
    Load and cache the language identification model.
    
    Returns:
        The loaded language identification model or None if loading fails
    """
    if not transformers_available:
        return None
        
    try:
        return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    except Exception as e:
        st.warning(f"Could not load advanced language detection model: {str(e)}")
        return None

def identify_language(text, language_identifier=None):
    """
    Identify the language of the given text.
    
    Args:
        text (str): Text to identify language for
        language_identifier: Pre-loaded language identification model (optional)
        
    Returns:
        tuple: (language_code, confidence_score)
    """
    # Handle empty text
    if not text or len(text.strip()) == 0:
        return 'en', 0.0
    
    # Fallback approach if langid is not available
    if not langid_available:
        # Very basic detection for common languages based on character frequency
        text = text.lower()
        
        # Check for characters common in specific languages
        if any(c in 'áéíóúüñ¿¡' for c in text):
            return 'es', 0.7  # Spanish with moderate confidence
        elif any(c in 'àâçéèêëîïôùûüÿ' for c in text):
            return 'fr', 0.7  # French
        elif any(c in 'äöüß' for c in text):
            return 'de', 0.7  # German
        elif any('\u4e00' <= c <= '\u9fff' for c in text):
            return 'zh', 0.8  # Chinese
        elif any('\u0400' <= c <= '\u04FF' for c in text):
            return 'ru', 0.8  # Russian
        elif any('\u0600' <= c <= '\u06FF' for c in text):
            return 'ar', 0.8  # Arabic
        elif any('\u0900' <= c <= '\u097F' for c in text):
            return 'hi', 0.8  # Hindi
        elif any('\u3040' <= c <= '\u30FF' for c in text):
            return 'ja', 0.8  # Japanese
        elif any('\uAC00' <= c <= '\uD7A3' for c in text):
            return 'ko', 0.8  # Korean
        else:
            return 'en', 0.5  # Default to English with low confidence
    
    try:
        # Use langid for basic detection if available
        lang, score = langid.classify(text)
        confidence = float(score)
        
        # For more accurate identification with transformer (if available)
        if language_identifier and len(text.split()) > 3:
            try:
                result = language_identifier(text)
                transformer_lang = result[0]['label']
                transformer_score = result[0]['score']
                # Only use transformer result if confidence is higher
                if transformer_score > confidence:
                    return transformer_lang, transformer_score
            except Exception:
                pass
        return lang, confidence
    except Exception as e:
        st.error(f"Language identification error: {str(e)}")
        # Last resort fallback
        return 'en', 0.1

def segment_by_language(text, language_identifier=None, min_segment_length=10):
    """
    Segment text into chunks by language.
    
    Args:
        text (str): Text to segment
        language_identifier: Pre-loaded language identification model (optional)
        min_segment_length (int): Minimum number of characters for a segment
        
    Returns:
        list: List of tuples containing (language_code, confidence, text_segment)
    """
    # Return empty list for empty text
    if not text or len(text.strip()) == 0:
        return []
    
    segments = []
    
    # Try to detect language boundaries using punctuation and paragraphs first
    import re
    
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    
    for paragraph in paragraphs:
        # Handle short paragraphs directly
        if len(paragraph) < min_segment_length:
            if paragraph.strip():  # Skip empty paragraphs
                lang, confidence = identify_language(paragraph, language_identifier)
                segments.append((lang, confidence, paragraph.strip()))
            continue
        
        # Split longer paragraphs into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_segment = []
        current_lang = None
        current_confidence = 0.0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Identify language of sentence
            if len(sentence) < min_segment_length:
                # For very short sentences, add to current segment to avoid over-segmentation
                if current_segment:
                    current_segment.append(sentence)
                    continue
                else:
                    # If no current segment, handle short sentence directly
                    lang, confidence = identify_language(sentence, language_identifier)
            else:
                lang, confidence = identify_language(sentence, language_identifier)
            
            if current_lang is None:
                current_lang = lang
                current_confidence = confidence
                current_segment = [sentence]
            elif lang == current_lang:
                # Same language, continue the segment
                current_segment.append(sentence)
            else:
                # Language changed, save current segment and start new one
                combined_segment = " ".join(current_segment)
                if combined_segment.strip():
                    segments.append((current_lang, current_confidence, combined_segment))
                current_segment = [sentence]
                current_lang = lang
                current_confidence = confidence
        
        # Add the last segment from this paragraph
        if current_segment:
            combined_segment = " ".join(current_segment)
            if combined_segment.strip():
                segments.append((current_lang, current_confidence, combined_segment))
    
    # Merge adjacent segments of the same language to avoid over-segmentation
    merged_segments = []
    current_lang = None
    current_segment = []
    current_confidence = 0.0
    
    for lang, confidence, segment in segments:
        if current_lang is None:
            current_lang = lang
            current_confidence = confidence
            current_segment = [segment]
        elif lang == current_lang:
            current_segment.append(segment)
            # Update confidence as weighted average
            current_confidence = (current_confidence * len("".join(current_segment[:-1])) + 
                                  confidence * len(segment)) / len("".join(current_segment))
        else:
            # Language changed, save current segment and start new one
            combined_segment = " ".join(current_segment)
            if combined_segment.strip():
                merged_segments.append((current_lang, current_confidence, combined_segment))
            current_segment = [segment]
            current_lang = lang
            current_confidence = confidence
    
    # Add the last merged segment
    if current_segment:
        combined_segment = " ".join(current_segment)
        if combined_segment.strip():
            merged_segments.append((current_lang, current_confidence, combined_segment))
    
    return merged_segments

def get_language_name(lang_code):
    """
    Convert ISO language code to full language name.
    
    Args:
        lang_code (str): ISO 639-1 language code
        
    Returns:
        str: Full language name
    """
    language_map = {
        'en': 'English', 
        'es': 'Spanish', 
        'fr': 'French', 
        'de': 'German', 
        'it': 'Italian', 
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'sv': 'Swedish',
        'da': 'Danish',
        'no': 'Norwegian',
        'fi': 'Finnish',
        'ru': 'Russian', 
        'uk': 'Ukrainian',
        'hi': 'Hindi', 
        'bn': 'Bengali',
        'ta': 'Tamil',
        'te': 'Telugu',
        'ml': 'Malayalam',
        'ur': 'Urdu',
        'ja': 'Japanese', 
        'zh': 'Chinese', 
        'ko': 'Korean',
        'ar': 'Arabic',
        'fa': 'Persian',
        'he': 'Hebrew',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'ms': 'Malay',
        'tr': 'Turkish',
        'el': 'Greek',
        'cs': 'Czech',
        'hu': 'Hungarian',
        'ro': 'Romanian',
        'bg': 'Bulgarian'
    }
    return language_map.get(lang_code, f'Unknown ({lang_code})')

def get_language_emoji(lang_code):
    """
    Get emoji flag for language.
    
    Args:
        lang_code (str): ISO 639-1 language code
        
    Returns:
        str: Flag emoji for the language's primary country
    """
    # Map of language codes to country flag emojis
    emoji_map = {
        'en': '🇬🇧',  # English - UK flag
        'es': '🇪🇸',  # Spanish
        'fr': '🇫🇷',  # French
        'de': '🇩🇪',  # German
        'it': '🇮🇹',  # Italian
        'pt': '🇵🇹',  # Portuguese
        'nl': '🇳🇱',  # Dutch
        'pl': '🇵🇱',  # Polish
        'sv': '🇸🇪',  # Swedish
        'da': '🇩🇰',  # Danish
        'no': '🇳🇴',  # Norwegian
        'fi': '🇫🇮',  # Finnish
        'ru': '🇷🇺',  # Russian
        'uk': '🇺🇦',  # Ukrainian
        'hi': '🇮🇳',  # Hindi
        'bn': '🇧🇩',  # Bengali
        'ta': '🇮🇳',  # Tamil
        'te': '🇮🇳',  # Telugu
        'ml': '🇮🇳',  # Malayalam
        'ur': '🇵🇰',  # Urdu
        'ja': '🇯🇵',  # Japanese
        'zh': '🇨🇳',  # Chinese
        'ko': '🇰🇷',  # Korean
        'ar': '🇸🇦',  # Arabic
        'fa': '🇮🇷',  # Persian
        'he': '🇮🇱',  # Hebrew
        'th': '🇹🇭',  # Thai
        'vi': '🇻🇳',  # Vietnamese
        'id': '🇮🇩',  # Indonesian
        'ms': '🇲🇾',  # Malay
        'tr': '🇹🇷',  # Turkish
        'el': '🇬🇷',  # Greek
        'cs': '🇨🇿',  # Czech
        'hu': '🇭🇺',  # Hungarian
        'ro': '🇷🇴',  # Romanian
        'bg': '🇧🇬'   # Bulgarian
    }
    return emoji_map.get(lang_code, '🌐')  # Globe emoji as fallback

# Load language identifier model at module initialization
language_identifier = None
if transformers_available:
    language_identifier = load_language_id_model()

def process_multilingual_text(text, analyze_sentiment=False, detect_emotions=False):
    """
    Process multilingual text, detecting languages and optionally analyzing sentiment.
    
    Args:
        text (str): Text to process
        analyze_sentiment (bool): Whether to analyze sentiment for each segment
        detect_emotions (bool): Whether to detect emotions for each segment
        
    Returns:
        list: List of dictionaries with language info and analysis for each segment
    """
    from nlp import analyze_sentiment, detect_emotion, load_emotion_model
    
    segments = segment_by_language(text, language_identifier)
    result = []
    
    # Load emotion model if needed
    emotion_model = None
    if detect_emotions and transformers_available:
        emotion_model = load_emotion_model()
    
    for lang_code, confidence, segment in segments:
        segment_info = {
            'text': segment,
            'language': {
                'code': lang_code,
                'name': get_language_name(lang_code),
                'emoji': get_language_emoji(lang_code),
                'confidence': round(confidence, 2)
            }
        }
        
        # Add sentiment analysis if requested
        if analyze_sentiment:
            sentiment_label, sentiment_score = analyze_sentiment(segment)
            segment_info['sentiment'] = {
                'label': sentiment_label,
                'score': round(sentiment_score, 2)
            }
        
        # Add emotion detection if requested
        if detect_emotions and emotion_model:
            emotion_label, emotion_score = detect_emotion(segment, emotion_model)
            segment_info['emotion'] = {
                'label': emotion_label,
                'score': round(emotion_score, 2)
            }
        
        result.append(segment_info)
    
    return result