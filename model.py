import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
import pickle

# load the csv file
news_dataset= pd.read_csv('train.csv')

# print the first 5 rows of the dataframe
news_dataset.head()

# replace the null values with empty string
news_dataset= news_dataset.fillna('')

# merge title and author for prediction process
# create a new column (content) to store the combined data
news_dataset['content']= news_dataset['author'] + ' ' + news_dataset['title']

# identify the dependent and independent variable
# seperate the data and label
# col: axis= 1, row: axis= 0 (by default is axis= 0)
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# stemming (reduce word to its root word)
port_stem = PorterStemmer()

def stemming(content):
  stemmed_content= re.sub('[^a-zA-Z]', ' ', content)
  stemmed_content= stemmed_content.lower()
  stemmed_content= stemmed_content.split()
  stemmed_content= [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content= ' '.join(stemmed_content)
  return stemmed_content

news_dataset['content']= news_dataset['content'].apply(stemming)

print(news_dataset['content'])

# seperate the data and the model
X= news_dataset['content'].values
Y= news_dataset['label'].values

# converting the tectual data to numeric data
vectorizer= TfidfVectorizer()
vectorizer.fit(X)
X= vectorizer.transform(X)

# split the dataset to training and test data
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, stratify= Y, random_state= 42)

# train the model
model= LogisticRegression()

# fit the model
model.fit(X_train, Y_train)

# make a pickle file of the model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))



