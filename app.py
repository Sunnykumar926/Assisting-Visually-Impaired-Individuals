import os
import io
import base64
import pytesseract 
from gtts import gTTS
import streamlit as st

from PIL import Image, ImageDraw 
from langchain.chains import LLMChain 
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage 
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# image to Base64 format
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def analyze_image(image, prompt):
    try:
        image_base64 = image_to_base64(image)
        message = HumanMessage(
            content=[
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': f'data:image/png;base64,{image_base64}'}
            ]
        )
        response = model.invoke([message])
        return response.content.strip()
    except Exception as e:
        return f'Error: {str(e)}'
    
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.getvalue()


def main():
    st.set_page_config(page_title='AI Assistive Tool', layout='wide', page_icon='🤖')

    st.title('AI Assistive Tool for visually Impaired 👩‍🦯')

    st.write("""
    This AI-powered tool assists visually impaired indivisuals by leveraging image analysis.
    It provides the following features:
    - **Scene Understanding** : Describe the content of uploaded images.
    - **Text-to-Speech Conversion** : Extracts and reads aloud text from images using OCR.
    - **Object & Obstacle Detection** : Identifies objects & obstacles for safe navigation.
    - **Personalized Assistance** : Offers task-specific guidance based on image content, like reading labels or recognizing items.
             
    Upload an image to get started and let AI help you understand and interact with our environment!
             
""")
    st.sidebar.header("📂 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])

    st.sidebar.header('🔧 Instructions')
    st.sidebar.write("""
    1. Upload an image.
    2. Choose an option below:
        - 🖼️ Describe Scene : Get a description of the image.
        - 📜 Extract Text : Extract text from the image.
        - 🚧 Detect Objects & Obstacles : Identify obstacles and highlight them.
        - 🛠️ Personalized Assistance : Get task-specific help.
    3. Results will be read ailoud for easy understanding.
""")

    if uploaded_file:
        if 'last_uploaded_file' in st.session_state and st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.extracted_text = None 
            st.session_state.summarized_text = None 
        
        st.session_state.last_uploaded_file = uploaded_file 
        image = Image.open(uploaded_file)

        st.markdown("""<style>
            .centered-image {
                display: flex;
                 margin-left: auto;
                 margin-right: auto;

                width: 500px;
            }
        </style>""", unsafe_allow_html=True)

        image_base64 = image_to_base64(image)
        st.markdown(f'<img src="data:image/png;base64,{image_base64}" class="centered-image"/>', unsafe_allow_html=True)

        def style_button(label, key, active_button_key):
            if 'active_button' not in st.session_state:
                st.session_state.active_button = None 

            color = 'green' if st.session_state.get('active_button') == active_button_key else 'dodgerblue'

            button_html = f"""
            <style>
                div[data-testid='stButton'] > button {{
                    background-color: {color} !important ;
                    color : white !important;
                    border: none !important;
                    padding: 20px 40px !important;
                    cursor: pointer !important;
                    border-radius: 10px;
                    fond-size: 18px !important;
                    font-weight: bold;
                    transition: all 0.3s ease;
                }}

                div[data-testid='stButton'] > button: hover {{
                    background-color: darkorange !important;
                    transform: scale(1.1)
                }}
            </style>
            """
            st.markdown(button_html, unsafe_allow_html=True)
            return st.button(label, key=key, help=f'Click to activate {label}')
        
        if style_button('🎬 Describe Scene', key='scene_description', active_button_key='scene_description'):
            st.session_state.active_button = "scene_description"
            with st.spinner('Generating scene description...'):
                scene_prompt = 'Describe this image briefly'
                scene_description = analyze_image(image, scene_prompt)
                st.write(scene_description)
                st.subheader('Scene Description')
                st.success(scene_description)
                st.audio(text_to_speech(scene_description), format='audio/mp3')

        

if __name__ == '__main__':
    main()
    