import os
import re

from django.http import HttpResponse
from django.shortcuts import render
import json
# Create your views here.
from collections import Counter
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import keras
import h5py
def talkshow(request):
    return render(request,"talkshow.html")
def fit_tokenizer():
    # tokenizer_model
    def cleanup(text):
        text = re.sub(r"[^a-z\?\! ]", " ", text.lower())
        text = ' '.join(text.split()).strip()
        return text

    df = pd.read_csv('sorted_data_acl.csv')
    for f in ['title', 'review_text']:
        df[f] = df[f].fillna('')
        df[f] = df[f].apply(cleanup)

    df['text'] = df['title'] + ' ' + df['review_text']
    max_len = df['text'].str.len().max()

    # Calculate the average length of the text
    avg_len = df['text'].str.len().mean()
    median_len = df['text'].str.len().median()
    print(f'Maximum length: {max_len}')
    print(f'Average length: {avg_len}')
    print(f'Median length: {median_len}')

    lens = df['text'].apply(len)
    sorted_lens = lens.sort_values()
    # Apply the function to the 'text' column
    df['text'] = df['text'].apply(lemmatize_text)
    y = df['sentiment'] = df['sentiment'].apply(lambda x: 0 if x[0] == 'p' else 1)
    df['rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
    from nltk.stem import WordNetLemmatizer
    num_words=30000
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(df['text'])
    return tokenizer
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Lemmatize each word
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    # Return the lemmatized text
    return ' '.join(lemmas)

tokenizer = fit_tokenizer()
print(os.getcwd())
file = h5py.File("mymodel.h5")
model = keras.models.load_model(file)

def answer(request):
    q = request.GET['q']
    txt = lemmatize_text(q)
    dd = tokenizer.texts_to_sequences([txt])
    h = model.predict(dd)
    if h:
        h = float(h[0][0])
    # 5星 0,0.25,0.5,0.75,1
    return HttpResponse(json.dumps({"code":100000,"text":f"The prediction score is %.3f"%h}),content_type='application/json')
