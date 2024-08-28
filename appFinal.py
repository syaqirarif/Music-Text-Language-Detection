import sys
import os

# Add the .venv\Scripts\ directory to the Python path
venv_scripts_dir = os.path.join(os.path.dirname(__file__), ".venv", "Scripts")
sys.path.append(venv_scripts_dir)

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.utils import shuffle
import lyricsgenius
from PIL import Image
from io import BytesIO
import requests
from utils import preprocess_lyrics



# Set up the Genius API client
genius = lyricsgenius.Genius(
    "UIm6tPDw7sWQSyXJffXV5nd6h42kqdmcHsh3zyVOMU_YPAo8ofFrk5QZekmtQFH1"
)

# Streamlit app configuration
st.set_page_config(
    page_title="LANGUAGE CLASSIFICATION ", page_icon="🎤", layout="wide"
)

# Header section
st.title("🎤 LANGUAGE CLASSIFICATION IN MUSIC LYRICS USING NAÏVE BAYES ALGORITHM")


# Function to read files and process sentences
def file2sentences(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        st.error(f"File {filename} not found.")
        return []

    replacements = [
        ("?", "."), ("!", "."),  ("»", ""), ("«", ""), (":", ""), (";", ""),("...", "."),
        ("…", "."), ("\n", "."), ("  ", " "),('"', ""), ("„", ""),("'", ""),
    ]

    for old, new in replacements:
        txt = txt.replace(old, new)

    txt = re.sub(r"\b\d+\b", "", txt)
    txt = re.sub(r"\b\w\b", "", txt)
    txt = re.sub(r"\s+", " ", txt)

    sentences = txt.split(".")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences


def load_datasets(filepaths):
    datasets = {}
    for language, filepath in filepaths.items():
        sentences = file2sentences(filepath)
        if sentences:
            datasets[language] = sentences
    return datasets


filepaths = {
    "English": "english.txt",
    "German": "german.txt",
    "Dutch": "dutch.txt",
    "French": "francais.txt",
    "Indonesia": "indo.txt"
}

datasets = load_datasets(filepaths)

# Ensure all datasets are non-empty
min_size = min(len(sentences) for sentences in datasets.values())

# Randomly sample min_size sentences from each dataset to balance them
np.random.seed(42)  # for reproducibility
sampled_data = {
    language: np.random.choice(sentences, min_size, replace=False)
    for language, sentences in datasets.items()
}

# Combine all sentences
X = np.array(
    [sentence for sentences in sampled_data.values() for sentence in sentences]
)
y = np.array(
    [language for language, sentences in sampled_data.items() for _ in sentences]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Create a TfidfVectorizer for character n-grams
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = Pipeline([("tfidf", tfidf), ("nb", MultinomialNB())])

# Define the parameter grid for GridSearchCV
param_grid = {"tfidf__ngram_range": [(2, 2), (2, 3)], "nb__alpha": [0.1, 1.0, 10.0]}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best parameters found by Grid Search
best_params = grid_search.best_params_

# Train the final model with the best parameters
final_model = grid_search.best_estimator_

# Shuffle the training data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Reevaluate the model
cm_resampled = confusion_matrix(y_test, final_model.predict(X_test))
accuracy_resampled = final_model.score(X_test, y_test)
logloss_resampled = log_loss(y_test, final_model.predict_proba(X_test), labels=list(final_model.classes_))

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
                            for verse, predicted_language in verse_results:
                                st.subheader(f"Verse in {predicted_language}")
                                st.text_area("Verse", value=verse, height=150)
                    else:
                        st.warning(
                            f"Album cover not found for '{song.title}' by '{song.artist}'"
                        )
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
                # Detect the language of the user text using langdetect
                user_text_processed = [user_text]  # Wrap the text in a list to process as a single instance
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

# Classification analysis section
with st.expander("Show Classification Analysis"):
    st.subheader("Best Parameters")
    st.write(best_params)

    # Confusion Matrix
    cm = confusion_matrix(y_test, final_model.predict(X_test))
    labels = list(datasets.keys())
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    # Additional Metrics
    accuracy = final_model.score(X_test, y_test)
    logloss = log_loss(y_test, final_model.predict_proba(X_test), labels=labels)

    # Precision, Recall, F1-score per class
    cr = classification_report(
        y_test, final_model.predict(X_test), target_names=labels, output_dict=True
    )
    df_cr = pd.DataFrame(cr).transpose()

    # Display additional metrics
    st.subheader("Additional Metrics")
    st.text(f"Accuracy: {accuracy:.4f}")
    st.text(f"Log Loss: {logloss:.4f}")

    # Calculate precision and recall for each class
y_scores = final_model.predict_proba(X_test)
precision = dict()
recall = dict()

for i, label in enumerate(final_model.classes_):
    precision[label], recall[label], _ = precision_recall_curve(y_test == label, y_scores[:, i])

# Create a dataframe for precision and recall
pr_data = []
for label in final_model.classes_:
    for p, r in zip(precision[label], recall[label]):
        pr_data.append({"Class": label, "Precision": p, "Recall": r})

pr_df = pd.DataFrame(pr_data)

# Plot precision-recall curves
fig_pr, ax_pr = plt.subplots()
sns.lineplot(data=pr_df, x="Recall", y="Precision", hue="Class", ax=ax_pr)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
st.pyplot(fig_pr)

# ROC Curve
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, label in enumerate(final_model.classes_):
    fpr[label], tpr[label], _ = roc_curve(y_test == label, y_scores[:, i])
    roc_auc[label] = auc(fpr[label], tpr[label])
    plt.plot(fpr[label], tpr[label], label=f'{label} (AUC = {roc_auc[label]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Display the plot using st.pyplot()
st.pyplot(fig_roc)

# Classification analysis section (optional)
with st.expander("Show Classification Analysis"):
    st.subheader("Best Parameters")
    st.write(best_params)

