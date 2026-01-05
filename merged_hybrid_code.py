import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np
import tempfile
from keras.applications.mobilenet_v3 import preprocess_input

# Chatbot imports
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# -------------------------------
# Tensorflow Model Loading
# -------------------------------
import requests

# -------------------------------
# TensorFlow Model Loading from Google Drive
# -------------------------------
MODEL_PATH = "Trained_Model.keras"  # Local file name
GDRIVE_FILE_ID = "1L0XAM8jjpHO_bUoSlWBRLSEbZEgdcs0U"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"  # Direct download link

@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive… this may take a moment.")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.write("Model downloaded successfully.")
    # Load model
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


def model_prediction(test_image_path):
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)  

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease + Chat"]
)

# -------------------------------
# Home Page
# -------------------------------
if app_mode == "Home":
    st.markdown("""
    ## **OCT Retinal Analysis Platform**

#### **Welcome to the Retinal OCT Analysis Platform**

**Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

##### **Why OCT Matters**
OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

---

#### **Key Features of the Platform**

- **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
- **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
- **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

---

#### **Understanding Retinal Diseases through OCT**

1. **Choroidal Neovascularization (CNV)**
   - Neovascular membrane with subretinal fluid
   
2. **Diabetic Macular Edema (DME)**
   - Retinal thickening with intraretinal fluid
   
3. **Drusen (Early AMD)**
   - Presence of multiple drusen deposits

4. **Normal Retina**
   - Preserved foveal contour, absence of fluid or edema

---

#### **About the Dataset**

Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
- **Normal**
- **CNV**
- **DME**
- **Drusen**

Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

---

#### **Get Started**

- **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
- **Explore Results**: View categorized scans and detailed diagnostic insights.
- **Learn More**: Dive deeper into the different retinal diseases and how OCT helps diagnose them.

---

#### **Contact Us**

Have questions or need assistance? [Contact our support team](#) for more information on how to use the platform or integrate it into your clinical practice.

    """)

# -------------------------------
# About Page
# -------------------------------
elif app_mode == "About":
    st.header("About the Dataset")
    st.markdown("""
    #### Overview  
    Retinal **Optical Coherence Tomography (OCT)** is an imaging technique used to capture high-resolution cross-sections of the retina.  
    Every year, over **30 million OCT scans** are performed worldwide. Interpreting these images manually takes significant time and effort.  

    ---
    #### Disease Categories  
    - **CNV (Choroidal Neovascularization):** Neovascular membrane with subretinal fluid  
    - **DME (Diabetic Macular Edema):** Retinal thickening with intraretinal fluid  
    - **Drusen (Early AMD):** Multiple drusen deposits visible in the retina  
    - **Normal Retina:** Preserved foveal contour, no edema or fluid  

    ---
    #### Dataset Structure  
    - **Total Images:** 84,495 OCT scans (JPEG format)  
    - **Categories:** `NORMAL`, `CNV`, `DME`, `DRUSEN`  
    - **Split into 3 folders:** `train`, `test`, `val`  
    - **Naming convention:** `(disease)-(randomized patient ID)-(image number)`  

    ---
    #### Data Sources  
    Images were collected from multiple medical centers worldwide, including:  
    - Shiley Eye Institute, UC San Diego  
    - California Retinal Research Foundation  
    - Medical Center Ophthalmology Associates  
    - Shanghai First People’s Hospital  
    - Beijing Tongren Eye Center  

    ---
    #### Quality Control  
    Each image passed a **3-tier verification process**:  
    1. **Initial screening** by trained students (excluded low-quality images).  
    2. **Independent grading** by four ophthalmologists.  
    3. **Final verification** by two senior retinal specialists (>20 years experience).  

    A validation subset of 993 scans was **double-checked** by two graders, with disagreements resolved by a senior retinal expert.  

    ---
    #### Purpose  
    This dataset enables the training and evaluation of AI models for automated OCT disease detection, supporting **faster, more accurate diagnoses** in ophthalmology.
    """)


# -------------------------------
# Integrated Page (Prediction + Chatbot)
# -------------------------------
elif app_mode == "Disease + Chat":
    st.header("OCT Disease Identification and Chat Assistance")
    test_image = st.file_uploader("Upload your OCT Image:")

    temp_file_path = None
    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name

    if st.button("Predict") and temp_file_path:
        with st.spinner("Analyzing OCT Image..."):
            result_index = model_prediction(temp_file_path)
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            prediction_result = class_name[result_index]

        # Save prediction in session state
        st.session_state['last_prediction'] = prediction_result

        st.success(f"Model Prediction: **{prediction_result}**")
        st.image(test_image)

        # Auto-start chatbot message
        auto_message = f"The scan suggests **{prediction_result}**. Would you like me to explain more?"
        st.session_state['message_history'].append({'role': 'assistant', 'content': auto_message})
        
    # -------------------------------
    # Chatbot Section
    # -------------------------------
    st.subheader("Chat with AI Assistant")

    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.text(message['content'])

    user_input = st.chat_input("Ask me about your result...")
    if user_input:
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.text(user_input)

        # Inject last prediction context
        context_message = ""
        if st.session_state['last_prediction']:
            context_message = f"(Note: The latest OCT prediction is {st.session_state['last_prediction']})"

        # Send full conversation to chatbot for better context
        conversation = [
            HumanMessage(content=m['content'])
            for m in st.session_state['message_history']
            if m['role'] == 'user'
        ]
        conversation.append(HumanMessage(content=user_input + " " + context_message))

        response = chatbot.invoke({'messages': conversation}, config=CONFIG)
        ai_message = response['messages'][-1].content

        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        with st.chat_message('assistant'):
            st.text(ai_message)

# elif app_mode == "Disease + Chat":
#     st.header("OCT Disease Identification + Chat Assistant")
#     test_image = st.file_uploader("Upload your OCT Image:")

#     temp_file_path = None
#     if test_image is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
#             tmp_file.write(test_image.read())
#             temp_file_path = tmp_file.name

#     if st.button("Predict") and temp_file_path:
#         with st.spinner("Analyzing OCT Image..."):
#             result_index = model_prediction(temp_file_path)
#             class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
#             prediction_result = class_name[result_index]

#         # Save prediction in session state
#         st.session_state['last_prediction'] = prediction_result

#         st.success(f"Model Prediction: **{prediction_result}**")
#         st.image(test_image)

#         # Auto-start chatbot message
#        # Auto-start chatbot message (only add once per prediction)
#         auto_message = f"The scan suggests **{prediction_result}**. Would you like me to explain more?"
#         if not st.session_state.get("auto_message_added", False):
#             st.session_state['message_history'].append({'role': 'assistant', 'content': auto_message})
#             st.session_state["auto_message_added"] = True


#         # ------------------- CHAT SECTION -------------------
#         st.subheader("Chat with AI Assistant")

#         # Show previous messages
#         for message in st.session_state['message_history']:
#             with st.chat_message(message['role']):
#                 st.text(message['content'])

#         # Input + Clear button aligned
#         col1, col2 = st.columns([6, 1])
#         with col1:
#             user_input = st.chat_input("Ask me about your result...")
#         with col2:
#             if st.button("🗑️ Clear"):
#                 st.session_state['message_history'] = []
#                 st.session_state['last_prediction'] = None
#                 st.session_state["auto_message_added"] = False
#                 st.rerun()

#         # Handle user query
#         if user_input:
#             st.session_state['message_history'].append({'role': 'user', 'content': user_input})
#             with st.chat_message('user'):
#                 st.text(user_input)

#             # Inject last prediction context
#             context_message = ""
#             if st.session_state['last_prediction']:
#                 context_message = f"(Note: The latest OCT prediction is {st.session_state['last_prediction']})"

#             # Send conversation to chatbot
#             conversation = [
#                 HumanMessage(content=m['content'])
#                 for m in st.session_state['message_history']
#                 if m['role'] == 'user'
#             ]
#             conversation.append(HumanMessage(content=user_input + " " + context_message))

#             response = chatbot.invoke({'messages': conversation}, config=CONFIG)
#             ai_message = response['messages'][-1].content

#             st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
#             with st.chat_message('assistant'):
#                 st.text(ai_message)

         
        
