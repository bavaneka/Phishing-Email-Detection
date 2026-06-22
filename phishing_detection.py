import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from scipy.sparse import hstack

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. Load Dataset
# =========================

print("Loading dataset...")

df = pd.read_csv("phishing_email.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# =========================
# 2. Clean Email Text
# =========================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

print("Cleaning text...")

df["clean_email"] = df["text_combined"].apply(clean_text)

# =========================
# 3. URL Count Feature
# =========================

def count_urls(text):

    urls = re.findall(
        r'http[s]?://\S+',
        str(text)
    )

    return len(urls)

df["url_count"] = df["text_combined"].apply(count_urls)

# =========================
# 4. Keyword Count Feature
# =========================

keywords = [
    "verify",
    "urgent",
    "account",
    "password",
    "bank",
    "login",
    "click",
    "security",
    "free",
    "winner"
]

def keyword_count(text):

    text = str(text).lower()

    count = 0

    for word in keywords:
        if word in text:
            count += 1

    return count

df["keyword_count"] = df["text_combined"].apply(keyword_count)

# =========================
# 5. Email Length Feature
# =========================

df["email_length"] = df["text_combined"].apply(len)

print("Feature Extraction Completed!")

# =========================
# 6. TF-IDF Vectorization
# =========================

print("Converting text to TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_text = vectorizer.fit_transform(
    df["clean_email"]
)

print("TF-IDF Completed!")

# =========================
# 7. Combine Features
# =========================

extra_features = df[
    [
        "url_count",
        "keyword_count",
        "email_length"
    ]
].values

X = hstack([
    X_text,
    extra_features
])

y = df["label"]

# =========================
# 8. Train-Test Split
# =========================

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-Test Split Completed!")

# =========================
# 9. Train Model
# =========================

print("Training Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model Training Completed!")

# =========================
# 10. Predictions
# =========================

y_pred = model.predict(X_test)

# =========================
# 11. Accuracy
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

# =========================
# 12. Confusion Matrix
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# =========================
# 13. Plot Confusion Matrix
# =========================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Phishing Email Detection - Confusion Matrix")

plt.show()