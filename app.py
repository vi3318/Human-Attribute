import streamlit as st
import google.generativeai as genai
import os
import PIL.Image
import toml

secrets = toml.load('.streamlit/secrets.toml')

# Set API Key for Google Gemini
google_api_key = st.secrets["google_api_key"]

# Configure the Google Gemini API
genai.configure(api_key=google_api_key)

# Load the Gemini Model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


# Function to analyze human attributes
def analyze_human_attributes(image):
    prompt = """
    You are an AI trained to analyze human attributes from images with high accuracy. 
    Carefully analyze the given image and return the following structured details:

    You have to return all results as you have the image, don't want any apologize or empty results.

    - **Gender** (Male/Female/Non-binary)
    - **Age Estimate** (e.g., 25 years)
    - **Ethnicity** (e.g., Asian, Caucasian, African, etc.)
    - **Mood** (e.g., Happy, Sad, Neutral, Excited)
    - **Facial Expression** (e.g., Smiling, Frowning, Neutral, etc.)
    - **Glasses** (Yes/No)
    - **Beard** (Yes/No)
    - **Hair Color** (e.g., Black, Blonde, Brown)
    - **Eye Color** (e.g., Blue, Green, Brown)
    - **Headwear** (Yes/No, specify type if applicable)
    - **Emotions Detected** (e.g., Joyful, Focused, Angry, etc.)
    - **Confidence Level** (Accuracy of prediction in percentage)
    """
    response = model.generate_content([prompt, image])
    return response.text.strip()


# Streamlit App
st.set_page_config(page_title="Human Attribute Detection", page_icon=":guardsman:", layout="wide")



st.markdown("<h1 style='text-align: left;'>Human Attribute Detection with Gemini</h1>", unsafe_allow_html=True)
st.markdown("""
    This app uses Google's Gemini model to analyze and detect various human attributes from uploaded images.
    Simply upload an image, and the model will return detailed insights such as gender, age, mood, and more.
    """, unsafe_allow_html=True)

# Custom Sidebar with Instructions
with st.sidebar:
    st.image("https://plus.unsplash.com/premium_photo-1664297939846-330cfd170bae?q=80&w=2955&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", use_container_width=True)  # Optional: Add a sidebar image
    st.write("---")
    st.header("Instructions")
    st.write("1. Upload a clear image of a person.")
    st.write("2. Wait for the analysis to complete.")
    st.write("3. The AI will provide insights into various human attributes.")

# Image Upload
uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)
    person_info = analyze_human_attributes(img)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write(person_info)


st.markdown("""
    <footer style='text-align: center; padding: 20px; font-size: 14px;'>
        <p>Powered by Streamlit and Google Gemini</p>
        <p>Developed by Vi Dharia 😊</p>
    </footer>
""", unsafe_allow_html=True)