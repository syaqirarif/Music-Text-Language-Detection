import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, log_loss, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle
import lyricsgenius

# Set up the Genius API client
genius = lyricsgenius.Genius(
    "UIm6tPDw7sWQSyXJffXV5nd6h42kqdmcHsh3zyVOMU_YPAo8ofFrk5QZekmtQFH1"
)

# Streamlit app configuration
st.set_page_config(
    page_title="Lyrics Fetcher & Language Classifier", page_icon="🎤", layout="wide"
)

# Header section
st.title("🎤 Lyrics Fetcher & Language Classifier")

# Function to read files and process sentences
def file2sentences(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        st.error(f"File {filename} not found.")
        return []

    replacements = [
        ("?", "."),
        ("!", "."),
        ("»", ""),
        ("«", ""),
        (":", ""),
        (";", ""),
        ("...", "."),
        ("…", "."),
        ("\n", "."),
        ("  ", " "),
        ('"', ""),
        ("„", ""),
        ("'", ""),
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

# Define file paths for English, German, and French datasets
filepaths = {
    "English": "english.txt",
    "German": "german.txt",
    "French": "francais.txt"
}

# Load and sample sentences from each dataset
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
st.header("Classification Analysis")

# Best Parameters
st.subheader("Best Parameters")
st.write(best_params)

# Confusion Matrix
st.subheader("Confusion Matrix")
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
st.subheader("Additional Metrics")
accuracy = final_model.score(X_test, y_test)
logloss = log_loss(y_test, final_model.predict_proba(X_test), labels=labels)
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Log Loss: {logloss:.4f}")

# Precision, Recall, F1-score per class
st.subheader("Precision, Recall, F1-score per class")
cr = classification_report(
    y_test, final_model.predict(X_test), target_names=labels, output_dict=True
)
df_cr = pd.DataFrame(cr).transpose()
st.dataframe(df_cr)

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

# Classification Report
st.subheader("Classification Report")
st.text(classification_report(y_test, final_model.predict(X_test), target_names=labels))
