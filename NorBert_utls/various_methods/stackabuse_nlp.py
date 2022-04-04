'''
https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
'''
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/annotaion_3000_012label.csv', usecols = ['text', 'label']) 

labels = data['label'].values
text = data['text'].values

processed_features = []
nltk.download('stopwords')
for sentence in range(len(text)):
    #all special characters
    processed_feature = re.sub(r'\W', ' ', str(text[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


X_train, X_test, Y_train, Y_test = train_test_split(processed_features, 
                                                             labels, 
                                                             test_size=0.2,
                                                             shuffle=True)

text_classifier = RandomForestClassifier(n_estimators = 120000, random_state = 0)
text_classifier.fit(X_train, Y_train)


predictions = text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(accuracy_score(Y_test, predictions))
