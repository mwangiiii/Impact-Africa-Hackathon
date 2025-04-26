import streamlit as st
import os
import tempfile
import time
import numpy as np
import wave
from PIL import Image, ImageDraw, ImageFont

# Display header and basic info
st.set_page_config(page_title="Real-Time Subtitle Generator", page_icon="🎙️")
st.title("SautiQwetu: An AI-powered Subtitle Generator for Deaf Students in Kenya")
st.write("This app converts speech into subtitles with multilingual support, emotion detection, .")

# Check package availability first
requirements = {
    "speech_recognition": False,
    "torch": False,
    "whisper": False,
    "transformers": False,
    "langid": False,
    "textblob": False,
    "pillow": False,
    "keybert": False,
}

# Try importing each package
try:
    import speech_recognition as sr
    requirements["speech_recognition"] = True
except ImportError:
    st.error("speech_recognition package not found. Please install it with: pip install SpeechRecognition")

try:
    import torch
    requirements["torch"] = True
except ImportError:
    st.error("torch package not found. Please install it with: pip install torch")

try:
    import whisper
    requirements["whisper"] = True
except ImportError:
    st.error("whisper package not found. Please install it with: pip install openai-whisper")

try:
    from transformers import pipeline
    requirements["transformers"] = True
except ImportError:
    st.error("transformers package not found. Please install it with: pip install transformers")

try:
    import langid
    requirements["langid"] = True
except ImportError:
    st.error("langid package not found. Please install it with: pip install langid")

try:
    from textblob import TextBlob
    requirements["textblob"] = True
except ImportError:
    st.error("textblob package not found. Please install it with: pip install textblob")

try:
    from PIL import Image, ImageDraw, ImageFont
    requirements["pillow"] = True
except ImportError:
    st.error("Pillow package not found. Please install it with: pip install Pillow")

try:
    from keybert import KeyBERT
    requirements["keybert"] = True
except ImportError:
    st.warning("keybert package not found. Keyword extraction will not be available.")

# Initialize models with caching
@st.cache_resource
def load_whisper_model(model_name, device):
    """Load and cache the Whisper speech recognition model."""
    try:
        return whisper.load_model(model_name, device=device)
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

@st.cache_resource
def load_emotion_model():
    """Load and cache the emotion detection model."""
    try:
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    except Exception as e:
        st.warning(f"Could not load emotion detection model: {str(e)}")
        return None

@st.cache_resource
def load_toxicity_model():
    """Load and cache the toxicity detection model."""
    try:
        return pipeline("text-classification", model="unitary/toxic-bert")
    except Exception as e:
        st.warning(f"Could not load toxicity model: {str(e)}")
        return None

@st.cache_resource
def load_keyword_model():
    """Load and cache the KeyBERT model for keyword extraction."""
    try:
        return KeyBERT()
    except Exception as e:
        st.warning(f"Could not load KeyBERT model: {str(e)}")
        return None

@st.cache_resource
def load_language_id_model():
    """Load and cache the language identification model."""
    try:
        return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    except Exception as e:
        st.warning(f"Could not load language detection model: {str(e)}")
        return None

# Language detection functions
def identify_language(text, language_identifier=None):
    """Identify the language of the given text."""
    if not text or len(text.strip()) == 0:
        return 'en', 0.0
    
    if not requirements["langid"]:
        # Basic detection for common languages
        text = text.lower()
        if any(c in 'áéíóúüñ¿¡' for c in text):
            return 'es', 0.7
        elif any(c in 'àâçéèêëîïôùûüÿ' for c in text):
            return 'fr', 0.7
        elif any(c in 'äöüß' for c in text):
            return 'de', 0.7
        elif any('\u4e00' <= c <= '\u9fff' for c in text):
            return 'zh', 0.8
        elif any('\u0400' <= c <= '\u04FF' for c in text):
            return 'ru', 0.8
        elif any('\u0600' <= c <= '\u06FF' for c in text):
            return 'ar', 0.8
        elif any('\u0900' <= c <= '\u097F' for c in text):
            return 'hi', 0.8
        elif any('\u3040' <= c <= '\u30FF' for c in text):
            return 'ja', 0.8
        elif any('\uAC00' <= c <= '\uD7A3' for c in text):
            return 'ko', 0.8
        else:
            return 'en', 0.5
    
    try:
        lang, score = langid.classify(text)
        confidence = float(score)
        
        if language_identifier and len(text.split()) > 3:
            try:
                result = language_identifier(text)
                transformer_lang = result[0]['label']
                transformer_score = result[0]['score']
                if transformer_score > confidence:
                    return transformer_lang, transformer_score
            except Exception:
                pass
        return lang, confidence
    except Exception as e:
        st.error(f"Language identification error: {str(e)}")
        return 'en', 0.1

def segment_by_language(text, language_identifier=None, min_segment_length=10):
    """Segment text into chunks by language."""
    if not text or len(text.strip()) == 0:
        return []
    
    import re
    segments = []
    paragraphs = re.split(r'\n\s*\n', text)
    
    for paragraph in paragraphs:
        if len(paragraph) < min_segment_length:
            if paragraph.strip():
                lang, confidence = identify_language(paragraph, language_identifier)
                segments.append((lang, confidence, paragraph.strip()))
            continue
        
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_segment = []
        current_lang = None
        current_confidence = 0.0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(sentence) < min_segment_length:
                if current_segment:
                    current_segment.append(sentence)
                    continue
                else:
                    lang, confidence = identify_language(sentence, language_identifier)
            else:
                lang, confidence = identify_language(sentence, language_identifier)
            
            if current_lang is None:
                current_lang = lang
                current_confidence = confidence
                current_segment = [sentence]
            elif lang == current_lang:
                current_segment.append(sentence)
            else:
                combined_segment = " ".join(current_segment)
                if combined_segment.strip():
                    segments.append((current_lang, current_confidence, combined_segment))
                current_segment = [sentence]
                current_lang = lang
                current_confidence = confidence
        
        if current_segment:
            combined_segment = " ".join(current_segment)
            if combined_segment.strip():
                segments.append((current_lang, current_confidence, combined_segment))
    
    # Merge adjacent segments of same language
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
            current_confidence = (current_confidence * len("".join(current_segment[:-1])) + 
                                  confidence * len(segment)) / len("".join(current_segment))
        else:
            combined_segment = " ".join(current_segment)
            if combined_segment.strip():
                merged_segments.append((current_lang, current_confidence, combined_segment))
            current_segment = [segment]
            current_lang = lang
            current_confidence = confidence
    
    if current_segment:
        combined_segment = " ".join(current_segment)
        if combined_segment.strip():
            merged_segments.append((current_lang, current_confidence, combined_segment))
    
    return merged_segments

def get_language_name(lang_code):
    """Convert ISO language code to full language name."""
    language_map = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 
        'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
        'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish',
        'ru': 'Russian', 'uk': 'Ukrainian', 'hi': 'Hindi', 'bn': 'Bengali',
        'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'ur': 'Urdu',
        'ja': 'Japanese', 'zh': 'Chinese', 'ko': 'Korean', 'ar': 'Arabic',
        'fa': 'Persian', 'he': 'Hebrew', 'th': 'Thai', 'vi': 'Vietnamese',
        'id': 'Indonesian', 'ms': 'Malay', 'tr': 'Turkish', 'el': 'Greek',
        'cs': 'Czech', 'hu': 'Hungarian', 'ro': 'Romanian', 'bg': 'Bulgarian'
    }
    return language_map.get(lang_code, f'Unknown ({lang_code})')

# NLP functions
def detect_emotion(text, emotion_model=None):
    """Detect emotion from text."""
    try:
        if emotion_model:
            result = emotion_model(text)[0]
            label = result['label']
            score = round(result['score'], 2)
            
            emoji = {
                "joy": "😄", "sadness": "😢", "anger": "😡", 
                "fear": "😨", "love": "❤", "surprise": "😲"
            }.get(label, "🙂")
            
            return f"{label.capitalize()} {emoji}", score
        elif requirements["textblob"]:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0.3:
                return "Joy 😄", polarity
            elif polarity < -0.3:
                return "Sadness 😢", abs(polarity)
            else:
                return "Neutral 🙂", 1 - abs(polarity)
        else:
            return "Unknown 🤔", 0.0
    except Exception as e:
        st.error(f"Emotion detection error: {str(e)}")
        return "Unknown 🤔", 0.0

def detect_urgency(text):
    """Detect urgency in text."""
    urgent_keywords = [
        "help", "emergency", "urgent", "danger", "asap",
        "immediately", "fire", "accident", "now", "crisis"
    ]
    found_keywords = [word for word in urgent_keywords if word in text.lower()]
    is_urgent = len(found_keywords) > 0
    return is_urgent, found_keywords

def analyze_sentiment(text):
    """Analyze text sentiment."""
    if not requirements["textblob"]:
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
    """Detect toxic content in text."""
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
    """Extract keywords from text."""
    if not kw_model:
        return []
        
    try:
        keywords = kw_model.extract_keywords(text, top_n=top_n)
        return keywords
    except Exception as e:
        st.error(f"Keyword extraction error: {str(e)}")
        return []

def transcribe_audio(audio_file, whisper_model):
    """Transcribe audio using Whisper model."""
    if not requirements["whisper"]:
        st.error("Cannot transcribe audio because whisper package is not installed.")
        return ""
        
    try:
        audio_array = whisper.load_audio(audio_file)
        result = whisper_model.transcribe(audio_array, fp16=False)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""

# Check device and display in sidebar
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: {device}")

# App mode selection
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Basic Subtitles", "Multilingual Subtitles", "Continuous Listening", "Sign Language Mode"]
)

# Language options
language_options = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese", 
    "Russian", "Hindi", "Japanese", "Chinese", "Arabic", "Korean"
]

# Sign language options (only shown in sign language mode)
sign_language_options = {
    "ASL": "American Sign Language",
    "BSL": "British Sign Language",
    "ISL": "International Sign Language",
    "Auslan": "Australian Sign Language"
}

# Sidebar for language selection (only shown in multilingual mode)
if app_mode != "Basic Subtitles":
    st.sidebar.subheader("Language Settings")
    primary_language = st.sidebar.selectbox("Primary Language", language_options, index=0)
    secondary_language = st.sidebar.selectbox("Secondary Language", language_options, index=1)

    # Model selection
    st.sidebar.subheader("Model Selection")
    model_options = ["Whisper Tiny", "Whisper Base", "Whisper Small"]
    selected_model = st.sidebar.selectbox("Speech Recognition Model", model_options, index=0)

    # Enable emotion detection
    enable_emotion = st.sidebar.checkbox("Enable Emotion Detection", value=True)
    
    # Enable toxicity detection
    enable_toxicity = st.sidebar.checkbox("Enable Toxicity Detection", value=False)
    
    # Enable keyword extraction
    enable_keywords = st.sidebar.checkbox("Enable Keyword Extraction", value=False)
    
    # Sign language settings
    if app_mode == "Sign Language Mode":
        st.sidebar.subheader("Sign Language Settings")
        sign_lang = st.sidebar.selectbox(
            "Sign Language", 
            options=list(sign_language_options.keys()),
            format_func=lambda x: sign_language_options[x]
        )

# Map selection to model size
model_size = {
    "Whisper Tiny": "tiny",
    "Whisper Base": "base", 
    "Whisper Small": "small"
}

# Load appropriate models based on mode
whisper_model = None
emotion_model = None
toxicity_model = None
keyword_model = None
language_identifier = None

if app_mode != "Basic Subtitles" and requirements["whisper"]:
    with st.spinner(f"Loading Whisper {model_size.get(selected_model, 'base')} model..."):
        try:
            whisper_model = load_whisper_model(model_size.get(selected_model, "base"), device)
            if whisper_model:
                st.success("Speech recognition model loaded!")
        except Exception as e:
            st.error(f"Error loading Whisper model: {str(e)}")

    if enable_emotion and requirements["transformers"]:
        with st.spinner("Loading emotion detection model..."):
            emotion_model = load_emotion_model()
            if emotion_model:
                st.success("Emotion detection model loaded!")

    if enable_toxicity and requirements["transformers"]:
        with st.spinner("Loading toxicity detection model..."):
            toxicity_model = load_toxicity_model()
            if toxicity_model:
                st.success("Toxicity detection model loaded!")

    if enable_keywords and requirements["keybert"]:
        with st.spinner("Loading keyword extraction model..."):
            keyword_model = load_keyword_model()
            if keyword_model:
                st.success("Keyword extraction model loaded!")

    if requirements["transformers"]:
        with st.spinner("Loading language identification model..."):
            language_identifier = load_language_id_model()
            if language_identifier:
                st.success("Language identification model loaded!")

# Define custom audio loading function
def load_audio_alternative(audio_file):
    """Load audio without requiring FFmpeg"""
    try:
        with wave.open(audio_file, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            frames = wf.readframes(n_frames)
            
            if sample_width == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
            else:
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels).mean(axis=1)
            
            return audio_data
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None

# Recording audio
def record_audio(duration=10):
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("Adjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        st.success(f"Recording for {duration} seconds... Please speak now!")
        
        try:
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            
            # For basic mode, return the audio directly
            if app_mode == "Basic Subtitles":
                return audio
            
            # For advanced modes, save audio to temporary file for Whisper processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio.get_wav_data())
                return temp_audio.name
        except Exception as e:
            st.error(f"Error recording audio: {str(e)}")
            return None

# Sign Language Functions
def get_asl_gloss(text):
    """Convert English text to ASL gloss notation"""
    text = text.lower()
    words = text.split()
    filtered = [w for w in words if w not in ['a', 'an', 'the', 'of', 'to', 'is', 'are', 'am']]
    return ' '.join(filtered).upper()

def create_sign_image(word, size=(300, 200)):
    """Create a simple image representation of a sign"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
        img = Image.new('RGB', size, color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except IOError:
            font = ImageFont.load_default()
        
        text_width = len(word) * 15
        draw.text((size[0]/2 - text_width/2, size[1]/2 - 20), f"{word}", fill=(0, 0, 0), font=font)
        draw.text((size[0]/2, size[1]/2 + 20), "👋", fill=(0, 0, 0), font=font)
        
        img.save(temp_img.name)
        return temp_img.name

def display_sign_language(text, container, sign_language="ASL"):
    """Display sign language representation in the provided container"""
    with container:
        st.subheader(f"{sign_language} Sign Language Translation")
        
        if sign_language == "ASL":
            sign_gloss = get_asl_gloss(text)
        else:
            sign_gloss = text.upper()
        
        st.write("Sign Language Gloss:")
        st.markdown(f"<div style='padding: 10px; background-color: #e6f7ff; border-radius: 5px; font-weight: bold;'>{sign_gloss}</div>", unsafe_allow_html=True)
        
        st.write("Visual Sign Representation:")
        signs = sign_gloss.split()
        
        for i in range(0, len(signs), 5):
            cols = st.columns(min(5, len(signs) - i))
            for j, col in enumerate(cols):
                if i+j < len(signs):
                    sign_img = create_sign_image(signs[i+j])
                    col.image(sign_img, caption=signs[i+j])
                    try:
                        os.unlink(sign_img)
                    except:
                        pass

# ========== MODE IMPLEMENTATIONS ==========

# Basic subtitles mode
if app_mode == "Basic Subtitles":
    st.write("Click the button below and start speaking. Your speech will be converted into subtitles.")
    
    if st.button("Start Listening 🎧"):
        audio = record_audio(duration=5)
        
        if audio:
            recognizer = sr.Recognizer()
            try:
                text = recognizer.recognize_google(audio)
                st.markdown(f"<h2 style='color: green;'>📝 Subtitle: {text}</h2>", unsafe_allow_html=True)
                
                # Basic sentiment analysis
                sentiment, score = analyze_sentiment(text)
                st.write(f"Sentiment: {sentiment} (confidence: {score:.2f})")
                
                # Urgency detection
                is_urgent, keywords = detect_urgency(text)
                if is_urgent:
                    st.warning(f"⚠️ Urgent content detected! Keywords: {', '.join(keywords)}")
                
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand your speech. Please try again.")
            except sr.RequestError:
                st.error("Could not connect to the recognition service. Please check your internet connection.")

# Multilingual subtitles mode
elif app_mode == "Multilingual Subtitles":
    if not requirements["whisper"]:
        st.error("Multilingual subtitles require the whisper package. Please install it first.")
        st.stop()
        
    st.write("Record speech with multiple languages. The system will identify language segments.")
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Recording Duration (seconds)", 5, 30, 10)
        if st.button("Start Recording 🎤"):
            # Record audio
            audio_file = record_audio(duration)
            
            if audio_file:
                with st.spinner("Processing speech..."):
                    try:
                        # Transcribe with Whisper
                        audio_array = load_audio_alternative(audio_file)
                        if audio_array is not None:
                            result = whisper_model.transcribe(audio_array, fp16=False)
                            transcript = result["text"]
                            
                            if transcript:
                                st.subheader("Original Transcription")
                                st.write(transcript)
                                
                                # Segment by language
                                with st.spinner("Identifying language segments..."):
                                    segments = segment_by_language(transcript, language_identifier)
                                
                                # Display segments with identified languages
                                st.subheader("Language Segments")
                                for lang, confidence, segment_text in segments:
                                    lang_name = get_language_name(lang)
                                    st.markdown(f"**Language: {lang_name} ({lang}) - Confidence: {confidence:.2f}**")
                                    
                                    # Emotion detection
                                    if enable_emotion and emotion_model:
                                        emotion, emo_confidence = detect_emotion(segment_text, emotion_model)
                                        st.markdown(f"*Emotion: {emotion} (confidence: {emo_confidence:.2f})*")
                                    
                                    # Toxicity detection
                                    if enable_toxicity and toxicity_model:
                                        is_toxic, tox_confidence = detect_toxicity(segment_text, toxicity_model)
                                        if is_toxic:
                                            st.error(f"⚠️ Toxic content detected! Confidence: {tox_confidence:.2f}")
                                    
                                    # Keyword extraction
                                    if enable_keywords and keyword_model:
                                        keywords = extract_keywords(segment_text, keyword_model)
                                        if keywords:
                                            st.markdown("*Keywords:*")
                                            st.write(", ".join([f"{k[0]} ({k[1]:.2f})" for k in keywords]))
                                    
                                    st.markdown(f"```{segment_text}```")
                            else:
                                st.error("Failed to transcribe audio. Please try again.")
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(audio_file)
                        except:
                            pass

# Continuous listening mode
elif app_mode == "Continuous Listening":
    if not requirements["whisper"]:
        st.error("Continuous listening requires the whisper package. Please install it first.")
        st.stop()
        
    st.write("Continuous mode will listen and transcribe continuously.")
    
    cont_col1, cont_col2 = st.columns(2)
    with cont_col1:
        start_button = st.button("Start Continuous Mode 🔄")
    with cont_col2:
        stop_button = st.button("Stop ⏹️")
    
    # Display area for continuous transcription
    transcript_container = st.empty()
    status_indicator = st.empty()
    
    if start_button and not stop_button:
        status_indicator.info("Starting continuous listening mode...")
        
        # Set session state to control the loop
        if 'continuous_running' not in st.session_state:
            st.session_state.continuous_running = True
        
        transcript_history = []
        
        while st.session_state.continuous_running and not stop_button:
            # Record short audio segments
            audio_file = record_audio(duration=5)
            
            if audio_file:
                try:
                    # Transcribe with Whisper
                    audio_array = load_audio_alternative(audio_file)
                    if audio_array is not None:
                        result = whisper_model.transcribe(audio_array, fp16=False)
                        transcript = result["text"]
                        
                        if transcript:
                            # Identify language
                            lang, confidence = identify_language(transcript, language_identifier)
                            lang_name = get_language_name(lang)
                            
                            # Get emotion if enabled
                            emotion_text = ""
                            if enable_emotion and emotion_model:
                                emotion, emo_confidence = detect_emotion(transcript, emotion_model)
                                emotion_text = f" | Emotion: {emotion} ({emo_confidence:.2f})"
                            
                            # Get toxicity if enabled
                            toxicity_text = ""
                            if enable_toxicity and toxicity_model:
                                is_toxic, tox_confidence = detect_toxicity(transcript, toxicity_model)
                                if is_toxic:
                                    toxicity_text = f" | ⚠️ TOXIC ({tox_confidence:.2f})"
                            
                            # Add to transcript history
                            timestamp = time.strftime("%H:%M:%S")
                            transcript_history.append((timestamp, lang_name, transcript, emotion_text, toxicity_text))
                            
                            # Display updated transcript history
                            html_output = "<div style='max-height: 400px; overflow-y: auto;'>"
                            for ts, lang, text, emotion, toxicity in transcript_history:
                                html_output += f"<div style='margin-bottom: 10px; padding: 8px; border-radius: 5px; background-color: #f5f5f5;'>"
                                html_output += f"<small>{ts} | Language: {lang}{emotion}{toxicity}</small>"
                                html_output += f"<p style='margin: 5px 0;'>{text}</p>"
                                html_output += "</div>"
                            html_output += "</div>"
                            
                            transcript_container.markdown(html_output, unsafe_allow_html=True)
                    
                except Exception as e:
                    status_indicator.error(f"Error: {str(e)}")
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
            
            # Check if stop button was pressed
            if stop_button:
                break
    
    if stop_button:
        status_indicator.warning("Stopping continuous listening...")
        st.session_state.continuous_running = False

# Sign Language Mode
elif app_mode == "Sign Language Mode":
    st.write("This mode converts speech to sign language visuals for the hearing impaired.")
    
    # Create containers for different elements
    control_container = st.container()
    subtitle_container = st.container()
    sign_container = st.container()
    
    with control_container:
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Recording Duration (seconds)", 5, 15, 5)
        with col2:
            start_signing = st.button("Start Recording for Sign Language 👋")
    
    # When the button is clicked
    if start_signing:
        # Record audio
        audio_file = record_audio(duration)
        
        if audio_file:
            with st.spinner("Transcribing speech..."):
                try:
                    # Use speech_recognition directly for sign language mode
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_file) as source:
                        audio_data = recognizer.record(source)
                    transcript = recognizer.recognize_google(audio_data)
                    
                    # Display subtitle
                    with subtitle_container:
                        st.subheader("Spoken Text")
                        st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>{transcript}</div>", unsafe_allow_html=True)
                        
                        # Add emotion if enabled
                        if enable_emotion and emotion_model:
                            emotion, confidence = detect_emotion(transcript, emotion_model)
                            st.markdown(f"<small>Detected emotion: {emotion} (confidence: {confidence:.2f})</small>", unsafe_allow_html=True)
                        
                        # Add toxicity warning if enabled
                        if enable_toxicity and toxicity_model:
                            is_toxic, tox_confidence = detect_toxicity(transcript, toxicity_model)
                            if is_toxic:
                                st.error(f"⚠️ Toxic content detected! Confidence: {tox_confidence:.2f}")
                    
                    # Convert to sign language and display
                    display_sign_language(transcript, sign_container, sign_language=sign_lang)
                    
                except sr.UnknownValueError:
                    st.error("Sorry, I could not understand your speech. Please try again.")
                except sr.RequestError:
                    st.error("Could not connect to the recognition service. Please check your internet connection.")
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(audio_file)
                    except:
                        pass

# Display installation instructions if requirements are missing
if not all(requirements.values()):
    st.error("Some required packages are missing. Please install them to enable all features.")
    