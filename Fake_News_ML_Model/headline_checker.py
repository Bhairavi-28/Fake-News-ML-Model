### Import libraries
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

### Loading the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("\t\t\t\t\t ===== Fake News Checker ===== \t\t\t\t\t")

while True:
    headline = input("\nEnter a headline (or type 'exit' to quit): ")
    if headline.lower() == "exit":
        break

    ### Transform headline to numbers
    headline_tfidf = vectorizer.transform([headline])
    ### Prediction
    prediction = model.predict(headline_tfidf)

    ### Output result
    if prediction[0] == 0:
        print("Fake News")
    else:
        print("True News")