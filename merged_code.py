import streamlit as st
import tensorflow as tf
import keras
from keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
from recommendation import cnv, dme, drusen, normal

# Chatbot imports
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# -------------------------------
# Tensorflow Model Prediction
# -------------------------------
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("Trained_Model (1).keras")
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)  # return index of max element

# -------------------------------
# Chatbot Config
# -------------------------------
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease Identification", "Chatbot"]
)

# -------------------------------
# Home Page
# -------------------------------
if app_mode == "Home":
    st.markdown("""
    ## **OCT Retinal Analysis Platform**

    Welcome to the Retinal OCT Analysis Platform.  
    Use the sidebar to navigate between pages:
    - Learn **About the Dataset**
    - Perform **Disease Identification**
    - Interact with the **AI Chatbot**
    """)

# -------------------------------
# About Page
# -------------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    Retinal OCT (Optical Coherence Tomography) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients.  
    Each year, millions of OCT scans are performed. Our dataset has **84,495 images** across four categories: CNV, DME, Drusen, and Normal.  
    """)

# -------------------------------
# Disease Identification Page
# -------------------------------
elif app_mode == "Disease Identification":
    st.header("OCT Disease Identification")
    test_image = st.file_uploader("Upload your OCT Image:")
    if test_image is not None:
        # Save to a temporary file and get its path
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name

    if st.button("Predict") and test_image is not None:
        with st.spinner("Please Wait.."):
            result_index = model_prediction(temp_file_path)
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        st.success(f"Model Prediction: **{class_name[result_index]}**")

        # Recommendation Section
        with st.expander("Learn More"):
            st.image(test_image)
            if result_index == 0:
                st.write("OCT scan showing *CNV with subretinal fluid.*")
                st.markdown(cnv)
            elif result_index == 1:
                st.write("OCT scan showing *DME with retinal thickening and intraretinal fluid.*")
                st.markdown(dme)
            elif result_index == 2:
                st.write("OCT scan showing *drusen deposits in early AMD.*")
                st.markdown(drusen)
            elif result_index == 3:
                st.write("OCT scan showing a *normal retina with preserved foveal contour.*")
                st.markdown(normal)

# -------------------------------
# Chatbot Page
# -------------------------------
elif app_mode == "Chatbot":
    st.header("AI Chat Assistant")
    
    # Load conversation history
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.text(message['content'])

    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Save user input
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.text(user_input)

        # Get chatbot response
        response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
        ai_message = response['messages'][-1].content

        # Save bot response
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        with st.chat_message('assistant'):
            st.text(ai_message)
