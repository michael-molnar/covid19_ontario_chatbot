import numpy as np
import pandas as pd
import pickle
import re
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load model
model = load_model('model.h5')

# Load tokenizers
input_tokenizer = pickle.load(open('input_tokenizer.pkl', 'rb'))
output_tokenizer = pickle.load(open('output_tokenizer.pkl', 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
# Set max_length
max_length = 6

# Get the responses 
responses = pd.read_csv('chatbot_responses.csv')

contraction_mappings = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", 
                       "could've": "could have", "couldn't": "could not", "didn't": "did not", 
                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", 
                       "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 
                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 
                       "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
                       "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 
                       "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 
                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", 
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                       "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 
                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", 
                       "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", 
                       "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                       "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", 
                       "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", 
                       "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", 
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                       "you'll've": "you will have", "you're": "you are", "you've": "you have"}
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
            "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 
            'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
            'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
            'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
            "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
def clean_text(text):
    """
    Performs all necessary cleaning operations on text input.
    """
    # Lowercase the text
    new_text = text.lower()
    # Remove special characters
    new_text = re.sub(r'\([^)]*\)?', '', new_text)
    new_text = re.sub('"', '', new_text)
    new_text = re.sub('\\?', '', new_text)
    # Expand contractions
    new_text = ' '.join([contraction_mappings[x] if x in contraction_mappings else x for x in new_text.split(' ')])
    # Remove 's 
    new_text = re.sub(r"'s\b", '', new_text)
    # Replace non-alphabetic characters with a space
    new_text = re.sub('[^a-zA-Z]', ' ', new_text)
    # Split the text into tokens and remove stopwords
    tokens = [word for word in new_text.split() if word not in stopwords]
    # Keep only tokens that are longer than one letter long 
    words = []
    for t in tokens:
        if len(t) > 1:
            # Lemmatize the words
            words.append(lemmatizer.lemmatize(t))
    # Return a rejoined string 
    return (' '.join(words).strip())

def int_to_word(integer, tokenizer):
    """
    Converting encoded integer tokens back into words.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    cleaned = clean_text(msg)
    words = word_tokenize(cleaned)
    tokens = input_tokenizer.texts_to_sequences(words)
    if [] in tokens:
        tokens = list(filter(None, tokens))
    if tokens:
        tokens = np.array(tokens).reshape(1, len(tokens))
        pad_tokens = pad_sequences(tokens, maxlen = max_length, padding = 'post')
        pred_classes = model.predict(pad_tokens)
        out_class = np.argmax(pred_classes) + 1
        category = int_to_word(out_class, output_tokenizer)

        cat_answers = responses[responses['Intent'] == category.title()]
        resp = ''
        for row in cat_answers.index:
            r = cat_answers['Zone'][row] + ":  " + cat_answers['Response'][row]
            resp += r
    else:
        resp = 'Sorry, can you rephrase that? I am just learning!'

    return resp

if __name__ == "__main__":
    app.run(debug=True)