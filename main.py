from flask import Flask, render_template, request
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# loading the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# reading the dataset
df = pd.read_json('file1.json')
print(df.head())

# performing basic preprocessing on data
stop_words = nlp.Defaults.stop_words
lemmatizer = nlp.get_pipe("lemmatizer")

def preprocess_text(text):
    # tokenizing the words using SpaCy
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    # returning the filtered tokens
    return tokens

# creating a column containing processed question column
df['processed_question'] = df['question'].apply(preprocess_text)

# defining a function to compute similarity between the user input and the processed question
def compute_similarity(user_input, processed_question):
    # creating an instance of tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # fitting the user input on the tfidf vectorizer
    tfidf_user_input = tfidf_vectorizer.fit_transform([user_input])
    # transforming the column processed question
    tfidf_processed_question = tfidf_vectorizer.transform([' '.join(processed_question)])
    # returning the cosine similarity between the user input and the question column
    return cosine_similarity(tfidf_user_input, tfidf_processed_question)[0][0]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.form['user_input']
    processed_user_input = preprocess_text(user_input)
    # compute similarity between the user's input and each question in the dataset
    df['similarity'] = df['processed_question'].apply(lambda x: compute_similarity(' '.join(processed_user_input), x))
    # return the answer to the most similar question
    most_similar_index = df['similarity'].idxmax()
    answer = df.loc[most_similar_index]['answer']
    return {'answer': answer}

if __name__ == '__main__':
    app.run('0.0.0.0',5000)
