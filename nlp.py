from transformers import pipeline
from textblob import TextBlob

# load model for efficiency
emotion_model =pipeline("text-classification", model="")