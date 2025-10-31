from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline", "").strip()

    if not headline:
        return jsonify({"Result": "No headline provided. Please enter a valid headline."}), 400

    headline_tfidf = vectorizer.transform([headline])
    prediction = model.predict(headline_tfidf)[0]
    result = "Fake News" if prediction == 0 else "Real News"

    return jsonify({"Result": result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
