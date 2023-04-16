# The-Black-Project
#Saarthi : Virtual Health Assistant


#Saarthi : "Because every Arjun in dilemma needs a Saarthi like Krishna who can show you the right path."

Frequently, individuals may be afflicted with a condition, yet they may not be aware of the appropriate actions to take initially. In such cases, they often panic and hastily attempt to reach out to a hospital or search their conditions in Google but it often leds to show the severe cases which are not correct, and the level of panic amond them increases.

#Objective: The objective of Saarthi is aimed to resolve this issue by providing comprehensive, step-by-step instructions for first aid, medications,reducing the gap between users and doctors,help users to maintain a healthy lifestyle.

#Methodology:
The model first loads the required libraries, including Spacy, JSON, Random, Pyttsx3, SpeechRecognition, Tensorflow, Keras, NumPy, and Pandas. It then loads the Spacy English model for lemmatization and tokenization. It reads in six JSON files (which we have made  by collecting data from different resources) and concatenates them to form the training data. It preprocesses the training data by extracting the relevant tags and tokenizing the questions using Spacy. It then creates a bag of words representation for each question in the training data and converts the output tags to one-hot encoded vectors. It then trains a neural network using Keras and saves the trained model to a file.

The model also includes a function to predict the intent of user inputs using the trained model. The function takes a user input and tokenizes it using Spacy. It then converts the tokenized input into a bag of words representation and passes it through the trained neural network. The function returns the predicted intent and the confidence score of the prediction.

The model also includes functions for speech recognition and text-to-speech conversion, which allows the user to interact with the model through speech.

#Results:
The model trains a neural network using the training data and achieves a reasonable accuracy on the test data. It also provides a function to predict the intent of user inputs and achieves reasonable accuracy on the test data and produces the resonable output. The model also includes functions for speech recognition and text-to-speech conversion, allowing the user to interact with the model through speech.
![image](https://user-images.githubusercontent.com/117035260/232274650-1a65abab-bb49-480e-814b-ef84d63cf850.png)
