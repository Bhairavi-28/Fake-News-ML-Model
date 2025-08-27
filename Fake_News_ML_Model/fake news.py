### This the first project for Athena Awards
### FAKE NEWS DETECTOR

### Import required Libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

### Dataset to be used
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
    ### Adding Label column
fake["label"] = 0
true["label"] = 1
    ### Combining true and false for the dataset 
df = pd.concat([fake, true])
df = df.sample(frac = 1).reset_index(drop = True)
df.to_csv("dataset.csv", index = False)
    ### The dataset file
data = pd.read_csv("dataset.csv")

### Splitting the dataset
X = data['text']
Y = data['label']

### Training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.697, random_state = 72)

### Transforming words to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.7)
vectorizer.fit_transform(X_train)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

### Applying Logistic Regression
### Training model
model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)

### Predictions 
Y_predict = model.predict(X_test_tfidf)

### Evaluation
### Confusion Matrix, Accuracy, Classification Report
print("Accuracy: ", accuracy_score(Y_test, Y_predict))
print("Confusion Matrix: \n", confusion_matrix(Y_test, Y_predict))
print("Classification Report: \n", classification_report(Y_test, Y_predict))

### Save the model
import joblib
joblib.dump(model, "fake_news_model.pkl")   # Save the trained model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl") # Save the vectorizer