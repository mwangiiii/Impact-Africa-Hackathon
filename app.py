import streamlit as st
import speech_recognition as sr

st.set_page_config(page_title="Real-Time Subtitle Generator", page_icon="🎙️")
st.title(" Real-Time Subtitle Generator for the Hearing Impaired")
st.write("Click the button below and start speaking. Your speech will be converted into subtitles in real-time.")

if st.button("Start Listening 🎧"):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Adjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        st.success("Listening... Please speak now!")

        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            st.markdown(f"<h2 style='color: green;'>📝 Subtitle: {text}</h2>", unsafe_allow_html=True)
        except sr.UnknownValueError:
            st.error(" Sorry, I could not understand your speech. Please try again.")
        except sr.RequestError:
            st.error("  Could not connect to the recognition service. Please check your internet connection.")
