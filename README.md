import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Updated file paths for your local machine
true_zip_path = r"C:\Users\HP\OneDrive\Documents\news_data\fakenews.zip.zip"
fake_zip_path = r"C:\Users\HP\OneDrive\Documents\news_data\truenews.zip.zip"

# Extract and read the CSVs from ZIP files
with zipfile.ZipFile(true_zip_path, 'r') as zip_ref:
    true_file_name = zip_ref.namelist()[0]
    true_news = pd.read_csv(zip_ref.open(true_file_name))

with zipfile.ZipFile(fake_zip_path, 'r') as zip_ref:
    fake_file_name = zip_ref.namelist()[0]
    fake_news = pd.read_csv(zip_ref.open(fake_file_name))

# Label and combine data
true_news['label'] = 'REAL'
fake_news['label'] = 'FAKE'
data = pd.concat([true_news, fake_news], ignore_index=True).dropna()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare features and labels
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Predict sample news
sample_news = ["The government just announced a new policy to improve education."]
sample_tfidf = tfidf.transform(sample_news)
print("\nPrediction for sample news:", model.predict(sample_tfidf))
