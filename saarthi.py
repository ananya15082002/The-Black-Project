import spacy
import json
import random
import pyttsx3
import speech_recognition as sr
from tensorflow import keras
from keras.models import Model,Sequential,load_model
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
import json
import numpy as np
import pandas as pd
import random
import pickle

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def spacy_tokenize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

# Load the JSON files and concatenate them
all_data = []
for file_name in ['file1.json', 'file2.json', 'file3.json', 'file4.json', 'file5.json', 'file6.json']:
    with open(file_name, 'r') as f:
        data = json.load(f)
        all_data += data # use list concatenation instead of dictionary indexing


# Select the intents to be used for training and concatenate them
words = []
labels = []
documents = []
ignore_words = ['?', '!']

for intent in all_data:
    if len(intent['tags']) == 0:
        tag = "unspecified"
    else:     
        ##Extracting only the first tags as they're the most relevant
        tag = intent['tags'][0]
        question = intent["question"]
        wrds = spacy_tokenize(question)
    
        words.extend(wrds)
        documents.append((wrds, tag))
            
        if tag not in labels:
            labels.append(tag)
            
words = [w.lemma_.lower() for w in nlp.tokenizer(" ".join(words)) if w.text.lower() not in ignore_words]
words = sorted(list(set(words)))

labels = sorted(list(set(labels)))

print(len(documents), "documents")
print(len(labels), "labels", labels)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(labels, open('labels.pkl','wb'))

training = []
out_empty = [0 for _ in range(len(labels))]
for doc in documents:
    bag = []
    
    pattern_words = doc[0]
    pattern_words = spacy_tokenize(" ".join(pattern_words))

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        

    output_row = out_empty[:]
    output_row[labels.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

model = Sequential()
model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(train_y[0]), activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

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

# Modified function to take user input and return predicted intent and confidence score
def predict_intent(user_input):
    predictions = classify_local(user_input)
    if len(predictions) > 0:
        intent = predictions[0][0]
        confidence_score = float(predictions[0][1])
        return intent, confidence_score
    else:
        return None, 0.0

user_input = input("Enter your question: ")
intent, confidence_score = classify_local(user_input)
print("Predicted intent:", intent)
print("Confidence score:", confidence_score)



# initialize text-to-speech engine
engine = pyttsx3.init()

# define a function to speak a given text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# define a function to recognize speech input from the user
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except:
        return None

# define a function to answer a health-related query
def answer_query(query):
    # loop through the diseases in the dataset
    for disease in data['diseases']:
        # check if the query matches the name of the disease
        if disease['name'].lower() in query.lower():
            # generate a random symptom or treatment
            response = random.choice(disease['symptoms'] + disease['treatments'])
            return f"The {disease['name']} can cause {response}."
    # if no matching disease is found, provide a general response
    return "I'm sorry, I'm not sure how to answer that."

# main loop for the virtual assistant
while True:
    # prompt the user to speak or type a query
    speak("Hi I am Saarthi your health assistant ! How can I help you?")
    input_type = recognize_speech()
    if input_type is None:
        speak("I'm sorry, I didn't catch that. Please type your query.")
        query = input()
    else:
        print(f"Query: {input_type}")
        speak(f"You said: {input_type}. Is that correct?")
        confirm = recognize_speech()
        if confirm is None:
            speak("I'm sorry, I didn't catch that.")
            query = None
        elif 'yes' in confirm.lower():
            query = input_type
        else:
            speak("Please type your query.")
            query = input()
    if query is not None:
        # answer the query and speak the response
        response = answer_query(query)
        print(f"Response: {response}")
        speak(response)
        # prompt the user to write down the response
        speak("Would you like me to write down the response?")
        confirm = recognize_speech()
        if confirm is None:
            speak("I'm sorry, I didn't catch that.")
        elif 'yes' in confirm.lower():
            speak("Please tell me the file name.")
            filename = recognize_speech()
            if filename is None:
                speak("I'm sorry, I didn't catch that.")
            else:
                filename = filename.replace(' ', '_')
                filename += '.txt'
                with open(filename, 'w') as f:
                    f.write(response)
                speak(f"The response has been written to {filename}.")
        else:
            speak("Okay, let me know if you need anything else.")
            
