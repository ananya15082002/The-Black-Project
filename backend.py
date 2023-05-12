import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

file_names = ['file1.json', 'file2.json', 'file3.json', 'file4.json', 'file5.json', 'file6.json']

dataset_path = []
for file_name in file_names:
    with open(file_name) as f:
        dataset_path.extend(json.load(f))


class Backend:
    def __init__(self, dataset_path):
        # loading the SpaCy English language model
        self.nlp = spacy.load('en_core_web_sm')
        
        # reading the dataset
        self.df = pd.read_json(dataset_path)
        
        # performing basic preprocessing on data
        self.stop_words = self.nlp.Defaults.stop_words
        self.lemmatizer = self.nlp.get_pipe("lemmatizer")
        
        # creating a column containing processed question column
        self.df['processed_question'] = self.df['question'].apply(self.preprocess_text)
        
        # creating an instance of tfidf vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        # fitting the processed question on the tfidf vectorizer
        self.tfidf_processed_question = self.tfidf_vectorizer.fit_transform([' '.join(question) for question in self.df['processed_question']])
        
    def preprocess_text(self, text):
        # tokenizing the words using SpaCy
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        # returning the filtered tokens
        return tokens
        
    def get_answer(self, user_input):
        # preprocess the user input
        processed_user_input = self.preprocess_text(user_input)
        # fitting the user input on the tfidf vectorizer
        tfidf_user_input = self.tfidf_vectorizer.transform([' '.join(processed_user_input)])
        # returning the cosine similarity between the user input and the question column
        similarities = cosine_similarity(tfidf_user_input, self.tfidf_processed_question)[0]
        # finding the most similar question index
        most_similar_index = similarities.argmax()
        # returning the answer to the most similar question
        return self.df.loc[most_similar_index]['answer']
