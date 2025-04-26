import streamlit as st

# Try to import required packages with fallback options
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    st.warning("transformers package not found. Emotion detection will use a simplified approach.")
    transformers_available = False

try:
    from textblob import TextBlob
    textblob_available = True
except ImportError:
    st.warning("textblob package not found. Basic sentiment analysis will not be available.")
    textblob_available = False

try:
    import whisper
    whisper_available = True
except ImportError:
    st.error("whisper package not found. Advanced speech recognition will not work.")
    whisper_available = False

# New imports for enhanced functionality
try:
    from keybert import KeyBERT
    keybert_available = True
except ImportError:
    st.warning("keybert package not found. Keyword extraction will not be available.")
    keybert_available = False

# Initialize models with caching
@st.cache_resource
def load_whisper_model(model_name, device):
    """
    Load and cache the Whisper speech recognition model.
    
    Args:
        model_name (str): Size of the Whisper model (tiny, base, small, etc.)
        device (str): Device to run the model on (cuda or cpu)
        
    Returns:
        The loaded Whisper model
    """
    if not whisper_available:
        st.error("Cannot load Whisper model because the package is not installed.")
        return None
        
    try:
        return whisper.load_model(model_name, device=device)
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

@st.cache_resource
def load_emotion_model():
    """
    Load and cache the emotion detection model.
    
    Returns:
        The loaded emotion detection model or None if loading fails
    """
    if not transformers_available:
        return None
        
    try:
        # Updated to use the new model specified in added code
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    except Exception as e:
        st.warning(f"Could not load emotion detection model: {str(e)}")
        return None

@st.cache_resource
def load_toxicity_model():
    """
    Load and cache the toxicity detection model.
    
    Returns:
        The loaded toxicity model or None if loading fails
    """
    if not transformers_available:
        return None
        
    try:
        return pipeline("text-classification", model="unitary/toxic-bert")
    except Exception as e:
        st.warning(f"Could not load toxicity model: {str(e)}")
        return None

@st.cache_resource
def load_keyword_model():
    """
    Load and cache the KeyBERT model for keyword extraction.
    
    Returns:
        The loaded KeyBERT model or None if loading fails
    """
    if not keybert_available:
        return None
        
    try:
        return KeyBERT()
    except Exception as e:
        st.warning(f"Could not load KeyBERT model: {str(e)}")
        return None

def detect_emotion(text, emotion_model=None):
    """
    Detects the emotion from the given text using a pre-trained model.
    
    Args:
        text (str): Text to analyze for emotion
        emotion_model: Pre-loaded emotion model (optional)
        
    Returns:
        tuple: (emotion_label_with_emoji, confidence_score)
    """
    try:
        if emotion_model:
            result = emotion_model(text)[0]  # Get top prediction
            label = result['label']  # e.g 'joy', 'anger', 'sadness'
            score = round(result['score'], 2)  # Confidence score, rounded for readability
            
            # Map emotions to emojis
            emoji = {
                "joy": "😄",
                "sadness": "😢",
                "anger": "😡",
                "fear": "😨",
                "love": "❤",
                "surprise": "😲"
            }.get(label, "🙂")  # Default emoji if no matching label
            
            return f"{label.capitalize()} {emoji}", score
        elif textblob_available:
            # Fallback to TextBlob sentiment if emotion detection model not available
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0.3:
                return "Joy 😄", polarity
            elif polarity < -0.3:
                return "Sadness 😢", abs(polarity)
            else:
                return "Neutral 🙂", 1 - abs(polarity)
        else:
            # Ultimate fallback
            return "Unknown 🤔", 0.0
    except Exception as e:
        st.error(f"Emotion detection error: {str(e)}")
        return "Unknown 🤔", 0.0

def detect_urgency(text):
    """
    Detects if the text contains urgent keywords.
    
    Args:
        text (str): Text to analyze for urgency
        
    Returns:
        tuple: (is_urgent, found_keywords)
    """
    urgent_keywords = [
        "help", "emergency", "urgent", "danger", "asap",
        "immediately", "fire", "accident", "now", "crisis"
    ]
    found_keywords = [word for word in urgent_keywords if word in text.lower()]
    is_urgent = len(found_keywords) > 0
    return is_urgent, found_keywords

def analyze_sentiment(text):
    """
    Analyzes the sentiment polarity of the given text.
    
    Args:
        text (str): Text to analyze for sentiment
        
    Returns:
        tuple: (sentiment_with_emoji, polarity_score)
    """
    if not textblob_available:
        return "Unknown 🤔", 0.0
        
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Positive 😊", polarity
        elif polarity < -0.1:
            return "Negative 😞", polarity
        else:
            return "Neutral 😐", polarity
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return "Unknown 🤔", 0.0

def detect_toxicity(text, toxicity_model=None):
    """
    Detects if the text contains toxic content.
    
    Args:
        text (str): Text to analyze for toxicity
        toxicity_model: Pre-loaded toxicity model (optional)
        
    Returns:
        tuple: (is_toxic, confidence_score)
    """
    if not toxicity_model:
        return False, 0.0
        
    try:
        result = toxicity_model(text)[0]
        is_toxic = result['label'] == 'toxic'
        confidence = round(result['score'], 2)
        return is_toxic, confidence
    except Exception as e:
        st.error(f"Toxicity detection error: {str(e)}")
        return False, 0.0

def extract_keywords(text, kw_model=None, top_n=5):
    """
    Extracts keywords from the given text.
    
    Args:
        text (str): Text to extract keywords from
        kw_model: Pre-loaded KeyBERT model (optional)
        top_n (int): Number of top keywords to return
        
    Returns:
        list: List of (keyword, score) tuples
    """
    if not kw_model:
        return []
        
    try:
        keywords = kw_model.extract_keywords(text, top_n=top_n)
        return keywords
    except Exception as e:
        st.error(f"Keyword extraction error: {str(e)}")
        return []

def transcribe_audio(audio_file, whisper_model):
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio_file (str): Path to the audio file
        whisper_model: Pre-loaded Whisper model
        
    Returns:
        str: Transcribed text
    """
    if not whisper_available:
        st.error("Cannot transcribe audio because whisper package is not installed.")
        return ""
        
    try:
        # Load audio and convert to the format Whisper expects
        audio_array = whisper.load_audio(audio_file)
        
        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_array, fp16=False)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""