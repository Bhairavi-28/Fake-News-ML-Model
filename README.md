# Fake-News-ML-Model

[![Athena Award Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Faward.athena.hackclub.com%2Fapi%2Fbadge)](https://award.athena.hackclub.com?utm_source=readme)

This is my first project for Athena Awards 
I built a machine learning model that can detect whether a news headline/article is fake or true.

____

About the Project
I trained a Logistic Regression model on a dataset of fake and true news articles. The text was transformed into numbers using TF-IDF Vectorizer (basically turning words into math). After training, the model reached ~98% accuracy (yes, I screamed when I saw it). You can now type in any headline, and the model will predict if it’s Fake News or True News.

How It Works
1. Train the model:
Combines Fake.csv and True.csv into one dataset.
Splits into training/testing sets.
Trains Logistic Regression on the TF-IDF features.
Evaluates with confusion matrix + classification report.

2. Save & Load:
Saves the trained model (fake_news_model.pkl) and vectorizer (tfidf_vectorizer.pkl).
Loads them in another script so you can just type headlines and test it out.

3. Predict:
Run the program.
Enter a headline.
Model says: Fake News or True News.

How to Run

Clone this repo and run the scripts:
# Train and save the model
python train_model.py  
# Run the headline checker
python fake_news_checker.py

(Use any dataset you like from Kaggle. Unfortunately the Dataset I used was more than 25 MB so I was unable to upload it.)

Future Ideas

Maybe add a web app or simple UI.
Deploy it online for friends to try.
Experiment with other ML models (Naive Bayes, Random Forest, maybe even transformers).

___

This is my first machine learning project.
It started as: “What if I try this?” and it actually worked. I’m keeping it as simple as possible because I love the first version. 
