import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Convert the input text into a sequence of tokens
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Truncate the token list to the maximum sequence length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    
    # Pad the sequence to match the input shape expected by the model
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    # Find the word corresponding to the predicted index
    predicted_word = {index: word for word, index in tokenizer.word_index.items()}.get(predicted_word_index)
    
    return predicted_word if predicted_word else None

# Streamlit app
st.title("Next Word Prediction With LSTM and Early Stopping")
input_text = st.text_input("Enter a sequence of the Words", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Adjust to match your model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)  # Add the missing comma here
    st.write(f"Next Word: {next_word}")
