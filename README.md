# Phishing Email Detection

A Machine Learning project that detects whether an email is **Phishing** or **Safe** using text analysis and feature engineering.

## Features

- Email text preprocessing
- URL extraction
- Keyword detection
- Email length analysis
- TF-IDF vectorization
- Random Forest classification
- Accuracy and confusion matrix evaluation

## Dataset

- Total Emails: 82,486
- Labels:
  - 0 = Safe
  - 1 = Phishing

> Note: The dataset is not included in this repository due to file size limitations.

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

## Results

- Accuracy: **98.45%**
- Confusion Matrix:

```text
[[7817  118]
 [ 138 8425]]
```

## Project Structure

```text
Phishing-Email-Detection
│
├── phishing_detection.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots
```

## Run the Project

```bash
pip install -r requirements.txt
python phishing_detection.py
```
