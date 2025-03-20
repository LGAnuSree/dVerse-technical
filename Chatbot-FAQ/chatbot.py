import streamlit as st
import json
import torch
from sentence_transformers import SentenceTransformer, util
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

faq_file = "Amazon_sagemaker_Faq.txt"

try:
    with open(faq_file, "r", encoding="utf-8") as file:
        faq_data = json.load(file)  # Convert JSON text into a Python list
except Exception as e:
    st.error(f"Error loading FAQ file: {e}")
    st.stop()

# Extract questions and answers
questions = [faq["question"] for faq in faq_data]
answers = [faq["answer"] for faq in faq_data]

# Load the sentence-transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Convert FAQ questions into vector embeddings
question_embeddings = model.encode(questions, convert_to_tensor=True)

def get_best_response(user_query):
    """Find the best answer for a given user query using NLP similarity."""
    user_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)
    best_match_index = torch.argmax(similarities)
    
    # Set a threshold to ensure meaningful responses
    if similarities[0, best_match_index] > 0.3:
        return answers[best_match_index]
    
    return "I'm sorry, I don't have an answer for that. Try typing a more relevant question!"

# Streamlit UI
st.title("💬 Amazon SageMaker FAQ Chatbot")
st.write("Ask me anything about Amazon SageMaker!")

# User input text box
user_input = st.text_input("Type your question:")

if user_input:
    response = get_best_response(user_input)
    st.write(f"**Chatbot:** {response}")
