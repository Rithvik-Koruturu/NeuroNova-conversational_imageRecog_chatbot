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

    # Function to process the uploaded images
    def input_image_setup(uploaded_files):
        image_parts = []
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                mime_type = uploaded_file.type
                image_parts.append({
                    "mime_type": mime_type,
                    "data": bytes_data
                })
            else:
                raise FileNotFoundError("No file uploaded")
        return image_parts

    # Function to get a response for text input using the gemini-pro model
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
    st.set_page_config(page_title="NeuroNova-Multi Image and Text Conversational Chatbot")

    st.header("NeuroNova-Multi Image and Text Conversational Chatbot")

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Multi-image upload functionality
    uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    image_contexts = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        # Predefined prompt for the chatbot
        image_chatbot_prompt = """
        You are an expert assignment solver capable of solving coding problems by analyzing images. 
        Your task is to examine the uploaded images, understand the coding problems depicted, and generate 
        corresponding Java code that is functional and free of bugs. Please ensure that your responses are 
        clear, concise, and provide explanations where necessary.
        """

        try:
            image_data = input_image_setup(uploaded_files)
            for img_data in image_data:
                image_context = get_gemini_image_response(image_chatbot_prompt, [img_data])
                image_contexts.append(image_context)

            # Display the analysis results for each image
            st.subheader("Image Analysis:")
            for idx, context in enumerate(image_contexts):
                st.write(f"Analysis for Image {idx + 1}: {context}")

        except FileNotFoundError as e:
            st.error(str(e))

    # Unified input field for both text and image-based queries
    user_input = st.text_input("Ask a question (about the images or anything else):")

    # Button to get a response
    if st.button("Get Response"):
        if user_input:
            # Get response considering image contexts
            combined_image_context = "\n".join(image_contexts) if image_contexts else None
            response = get_gemini_text_response(user_input, combined_image_context)
            
            # Display response
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("You", user_input))
                st.session_state['chat_history'].append(("Bot", chunk.text))

    # Java code generation based on user input and image context
    if st.button("Generate Java Code"):
        if uploaded_files and user_input:
            java_code = f"""
            import java.io.*;

            public class ImageAnalysis {{

                public static void main(String[] args) {{
                    System.out.println("Image Analysis Results:");
                    System.out.println("{image_contexts}");
                    
                    // Sample question analysis
                    System.out.println("Question Analysis: {user_input}");
                }}
            }}
            """
            st.code(java_code, language='java')

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")
