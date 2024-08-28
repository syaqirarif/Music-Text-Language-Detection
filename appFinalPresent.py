import sys
import os

# Add the .venv\Scripts\ directory to the Python path (if necessary)
venv_scripts_dir = os.path.join(os.path.dirname(__file__), ".venv", "Scripts")
sys.path.append(venv_scripts_dir)

import streamlit as st
import lyricsgenius
from PIL import Image
from io import BytesIO
import requests
from langdetect import detect, LangDetectException
from langdetect.lang_detect_exception import ErrorCode

# Languages to detect
LANGUAGES = {'id': 'Indonesian', 'fr': 'French', 'de': 'German', 'nl': 'Dutch', 'en': 'English'}

def preprocess_lyrics(lyrics):
    verses = lyrics.split('\n\n')  # Assuming verses are separated by double newlines

    results = []
    for verse in verses:
        # Remove "(ContributorsTranslations)" if present
        verse = verse.replace("(ContributorsTranslations)", "")
        
        if verse.strip():
            try:
                detected_lang = detect(verse)
                if detected_lang in LANGUAGES:
                    predicted_language = LANGUAGES[detected_lang]
                    results.append((verse, predicted_language))
                else:
                    results.append((verse, "Unknown"))
            except LangDetectException as e:
                results.append((verse, "Unknown"))

    return results

# Set up the Genius API client
genius = lyricsgenius.Genius("UIm6tPDw7sWQSyXJffXV5nd6h42kqdmcHsh3zyVOMU_YPAo8ofFrk5QZekmtQFH1")

# Streamlit app configuration
st.set_page_config(page_title="LANGUAGE CLASSIFICATION", page_icon="🎤", layout="wide")

# Header section
st.title("🎤 LANGUAGE CLASSIFICATION IN MUSIC LYRICS USING NAÏVE BAYES ALGORITHM")

# Main content section
with st.container():
    st.write("Enter the artist name and song title to fetch album cover and lyrics.")

    # Input artist name
    artist_name = st.text_input("Artist Name", placeholder="e.g., Usher")

    # Input song title
    song_title = st.text_input("Song Title", placeholder="e.g., My Boo")

    # Fetch lyrics
    if st.button("Get Lyrics"):
        if artist_name.strip() and song_title.strip():
            try:
                song = genius.search_song(song_title, artist_name)
                if song:
                    st.subheader(f"Lyrics for {song.title} by {song.artist}")

                    # Display album cover
                    cover_url = song.song_art_image_url
                    if cover_url:
                        response = requests.get(cover_url)
                        img = Image.open(BytesIO(response.content))

                        # Display album cover and lyrics side by side
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(
                                img,
                                caption=f"Album Cover for {song.title} by {song.artist}",
                                width=200,
                            )
                        with col2:
                            lyrics = song.lyrics

                            # Process each verse and display results
                            verse_results = preprocess_lyrics(lyrics)
                            for i, (verse, predicted_language) in enumerate(verse_results):
                                st.subheader(f"Verse in {predicted_language}")
                                st.text_area("Verse", value=verse, height=150, key=f'verse_{i}')
                    else:
                        st.warning(f"Album cover not found for '{song.title}' by '{song.artist}'")
                else:
                    st.error(f"Lyrics for '{song_title}' by '{artist_name}' not found.")
            except Exception as e:
                st.error(f"Error fetching lyrics: {e}")
        else:
            st.warning("Please enter both artist name and song title.")

# User input for custom text classification
with st.container():
    st.write("Enter your own text to classify its language.")

    # Input user text
    user_text = st.text_area("Your Text", placeholder="Enter your text here...", height=150)

    # Classify the user text
    if st.button("Classify Text"):
        if user_text.strip():
            try:
                verse_results = preprocess_lyrics(user_text)
                if verse_results:
                    verse, predicted_language = verse_results[0]
                    st.write(f"The predicted language of the entered text is: {predicted_language}")
                else:
                    st.write("Unable to detect the language of the entered text.")
            except Exception as e:
                st.error(f"Error classifying text: {e}")
        else:
            st.warning("Please enter some text to classify.")
