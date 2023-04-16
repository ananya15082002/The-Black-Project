from flask import Flask, render_template, request
import spacy
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))
model = load_model('chatbot_model.hdf5')

def spacy_tokenize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

def clean_up_sentence(sentence):
    doc = nlp(sentence)
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input']).to_numpy()
    results = model.predict([input_data])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((labels[r[0]], str(r[1])))

    return return_list

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    bot_response = classify_local(user_text)
    return str(bot_response)

if __name__ == "__main__":
    app.run()
