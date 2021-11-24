import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords

with open('model.pkl','rb') as f:

    md = pickle.load(f)

with open('vector.pkl','rb') as f:
    vector = pickle.load(f)

st.title('Spam-Filter')
input = st.text_area('Enter text')


def tranform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if st.button('Check'):
    transformed_text = tranform_text(input)

    vectorized = vector.transform([transformed_text])
    result = md.predict(vectorized)[0]

    if result == 1:
        st.header('Not Spam')

    else:
        st.header('Spam')



