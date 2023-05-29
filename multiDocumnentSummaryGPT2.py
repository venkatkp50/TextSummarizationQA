import gpt_2_simple as gpt2clea
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import torch

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def preprocess_documents(documents):
    # Combine the documents into a single string
    text = ' '.join(documents)
    st.write(len(text))
    # Tokenize the text
    inputs = tokenizer.encode(text, return_tensors='pt')
    return inputs

def generate_summary(inputs, max_length=3):
    # Generate the summary using the GPT-2 model
    st.write('1.........',inputs)
    st.write('1.........',inputs.shape)
    summary_ids = model.generate(inputs, max_length=max_length, num_return_sequences=1, early_stopping=True)
    st.write('2.........')
    # Decode the summary tokens into text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.write('3.........')
    return summary



# Preprocess the documents
# inputs = preprocess_documents(documents)

# # Generate the summary
# summary = generate_summary(inputs)

# # Print the summary
# print(summary)