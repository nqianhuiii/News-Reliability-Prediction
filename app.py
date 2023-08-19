import numpy as np
from flask import Flask, request, render_template
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
nltk.download('stopwords')
import string

# create flask app
app = Flask(__name__, static_url_path='/static')

# load the pickle model
model= pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# preprocess the input text
port_stem = PorterStemmer()
def preprocess_text(string_features):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', string_features)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods= ["post"])
def prediction():
    global user_input
    string_features = list(request.form.values())  # convert dictionary values to a list of strings
    user_input= string_features[0]
    preprocessed_text = [preprocess_text(text) for text in string_features]  # preprocess each string
    X = vectorizer.transform(preprocessed_text)
    prediction = model.predict(X)

    prob_scores = model.predict_proba(X)[0]  # get predicted probability scores
    reliable_percentage = prob_scores[1] * 100  # percentage of reliable news
    return render_template('index.html', prediction_result=f"The news is {reliable_percentage:.2f}% reliability", user_input= user_input)


if __name__=="__main__":
    app.run(debug= True)