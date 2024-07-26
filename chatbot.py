import pickle
import nltk
import numpy as np
import json
from keras.models import load_model
import random
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data files
intents_file = json.loads(open('intents_file.json').read())
lem_words = pickle.load(open('lem_words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
bot_model = load_model('chatbot_model.keras')


# Function to clean and tokenize input text
def cleaning(text_in):
    words = nltk.word_tokenize(text_in)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words


# Function to convert text into a bag of words
def bag_ow(text_in, words, show_details=True):
    sentence_words = cleaning(text_in)
    bag_of_words = [0] * len(words)
    for s in sentence_words:
        for ind, w in enumerate(words):
            if w == s:
                bag_of_words[ind] = 1
    return np.array(bag_of_words)


# Function to predict the class of the input sentence
def class_prediction(sentence, model):
    p = bag_ow(sentence, lem_words, show_details=False)
    result = model.predict(np.array([p]))[0]
    ER_THRESHOLD = 0.30
    f_results = [[i, r] for i, r in enumerate(result) if r > ER_THRESHOLD]
    f_results.sort(key=lambda x: x[1], reverse=True)
    intent_prob_list = []
    for i in f_results:
        intent_prob_list.append({"intent": classes[i[0]], "probability": str(i[1])})
    return intent_prob_list


# Function to get a response from the bot based on the predicted intent
def get_bot_response(ints, intents):
    tag = ints[0]['intent']
    intents_list = intents['intents']
    for intent in intents_list:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    return f"MountieBot: {result}"


# Main function to get the bot response
def bot_response(text_input):
    ints = class_prediction(text_input, bot_model)
    response = get_bot_response(ints, intents_file)
    return response


# Chat loop
for i in range(5):
    text = input("You : ")
    print(bot_response(text))
