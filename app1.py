from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
from pathlib import Path

# Load environment variables (like your Google API key)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is set; if not, display an error message
if not api_key:
    st.error("API key is not set. Please check your .env file.")
else:
    # Configure the generative AI model with the provided API key
    genai.configure(api_key=api_key)

    # Function to get the response from the generative AI model for images
    def get_gemini_image_response(input_prompt, image_data=None):
        model = genai.GenerativeModel('gemini-1.5-flash')
        if image_data:
            response = model.generate_content([image_data[0], input_prompt])
            return response.text
        else:
            return "No image data provided."

    # Function to process the uploaded image
    def input_image_setup(uploaded_file):
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            mime_type = uploaded_file.type
            image_parts = [
                {
                    "mime_type": mime_type,
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")

    # Function to get response for text input using the gemini-pro model
    def get_gemini_text_response(question, image_context=None):
        # Load Gemini Pro model
        model = genai.GenerativeModel("gemini-pro") 
        chat = model.start_chat(history=[])
        
        # Include image context if available
        if image_context:
            combined_prompt = f"Context: {image_context}\nQuestion: {question}"
            response = chat.send_message(combined_prompt, stream=True)
        else:
            response = chat.send_message(question, stream=True)
            
        return response

    # Initialize Streamlit app
    st.set_page_config(page_title="NeuroNova-Image and Text Conversational Chatbot")

    st.header("NeuroNova-Image and Text Conversational Chatbot")

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Image upload functionality
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image_context = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Predefined prompt for the chatbot
        image_chatbot_prompt = """
        You are an advanced AI agent capable of combining natural language processing and image recognition technology. 
        Your task is to recognize objects in an image, describe them, and answer questions based on the image content. 
        You utilize a neural encoder-decoder model with a Late Fusion encoder, enabling you to interpret both image and text inputs.
        """

        try:
            image_data = input_image_setup(uploaded_file)
            image_context = get_gemini_image_response(image_chatbot_prompt, image_data)
            
            # Display the analysis results
            st.subheader("Image Analysis:")
            st.write(image_context)

        except FileNotFoundError as e:
            st.error(str(e))

    # Unified input field for both text and image-based queries
    user_input = st.text_input("Ask a question (about the image or anything else):")

    # Button to get a response
    if st.button("Get Response"):
        if user_input:
            if image_context:
                # Get response considering image context
                response = get_gemini_text_response(user_input, image_context)
            else:
                # Get general text response
                response = get_gemini_text_response(user_input)
            
            # Display response
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("You", user_input))
                st.session_state['chat_history'].append(("Bot", chunk.text))

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

    # Additional sidebar functionalities
    st.sidebar.markdown("© 2024 Team NeuroNova")
    st.sidebar.markdown("- Rithvik K")
    st.sidebar.markdown("- Uttej Kumar O")
    st.sidebar.markdown("- Sudheera K")
    st.sidebar.markdown("- Santharam S")
    st.sidebar.markdown("- Kasyap G")
    st.sidebar.markdown("- Yogender")
