import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example dataset
data = [
    ("Your account has been compromised, click here to reset", 1),  # Phishing
    ("Congratulations, you won a prize! Enter your details here", 1),
    ("Please update your payment information to avoid suspension", 1),
    ("Join us for lunch today at 1pm!", 0),
    ("Team meeting postponed to Friday", 0),
    ("Check out the project documentation", 0),
]

df = pd.DataFrame(data, columns=["message", "label"])

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["message"])
y = df["label"]

# Train classifier
model = LogisticRegression()
model.fit(X, y)

# Test messages
test_messages = [
    "Verify your account urgently to avoid deactivation",
    "Your latest assignment is uploaded to the portal",
]

X_test = tfidf.transform(test_messages)
predictions = model.predict(X_test)

for msg, pred in zip(test_messages, predictions):
    print(f"Phishing detected: {bool(pred)}\tMessage: \"{msg}\"")