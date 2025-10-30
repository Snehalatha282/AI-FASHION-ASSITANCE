import streamlit as st
import numpy as np
import PIL.Image
import io
import requests
import google.generativeai as genai
import base64
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings
import os
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress ALTS credentials warning
os.environ['GRPC_TRACE'] = 'none'
os.environ['GRPC_VERBOSITY'] = 'none'
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")

# API key validation and instructions
if not api_key:
    st.error("""
    API Key not found! Please follow these steps:
    
    1. Go to https://makersuite.google.com/app/apikey
    2. Sign in with your Google account
    3. Click 'Create API key' or use an existing key
    4. Copy the API key
    5. Create or edit the .env file in your project directory
    6. Add this line to your .env file:
       GOOGLE_API_KEY=your_api_key_here
    7. Replace 'your_api_key_here' with your actual API key
    8. Restart the Streamlit app
    """)
    st.stop()

try:
    # Test the API key with a simple generation
    genai.configure(api_key=api_key)
    # List available models to debug
    try:
        models = genai.list_models()
        st.write("Available models:", [m.name for m in models])
        
        # Use gemini-pro-latest for initial test
        model = genai.GenerativeModel('models/gemini-pro-latest')
        response = model.generate_content("Test")
        response.resolve()
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
        st.stop()
except Exception as e:
    st.error(f"""
    Invalid API key! Please check:
    1. You've copied the entire API key correctly
    2. You've enabled the Gemini API in your Google Cloud project
    3. Your API key is properly formatted in the .env file
    
    Error details: {str(e)}
    """)
    st.stop()

def format_image_input(data):
    # Fix: get two different image paths if available and convert to absolute paths
    uris = data.get("uris", [[]])
    image_path = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(uris[0]) > 0:
        path = os.path.join(base_dir, uris[0][0].replace('\\', '/'))
        image_path.append(path)
    if len(uris[0]) > 1:
        path = os.path.join(base_dir, uris[0][1].replace('\\', '/'))
        image_path.append(path)
    elif len(uris) > 1 and len(uris[1]) > 0:
        path = os.path.join(base_dir, uris[1][0].replace('\\', '/'))
        image_path.append(path)
    return image_path

def open_image(img_data):
    if isinstance(img_data, str):
        try:
            # Try to open as local file first
            img = PIL.Image.open(img_data)
        except Exception as e:
            # If local file fails, try as URL
            try:
                response = requests.get(img_data)
                img = PIL.Image.open(io.BytesIO(response.content))
            except Exception as url_e:
                raise ValueError(f"Could not open image from path or URL: {img_data}. Error: {str(url_e)}")
    elif isinstance(img_data, np.ndarray):
        img = PIL.Image.fromarray(img_data.astype('uint8'))
    elif isinstance(img_data, list):
        try:
            img_data = np.array(img_data, dtype='uint8')
            img = PIL.Image.fromarray(img_data)
        except Exception as e:
            st.error(f"Error converting list to array : {e}")
            raise ValueError("Unsupported image data format")
    else:
        raise ValueError("Unsupported image data format")
    return img

# Helper to convert PIL image to bytes for Gemini
def pil_image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

st.title("AI the Fashion Styling Assistant")
st.title("Enter your styling query and get image-based recommendations, or upload an image to retrieve similar images.")

uploaded_file = st.file_uploader("upload an image to retrieve similar images:", type=["jpg","jpeg","png"])
query = st.text_input("Or, enter your styling query:")


if st.button("Generate Styling Ideas / Retrieve Styling Ideas / Retrieve Images"):
    chroma_client = chromadb.PersistentClient(path="Vector_database")
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

    # Initialize the vision model for image analysis
    try:
        model = genai.GenerativeModel('models/gemini-2.5-pro')  # Using the latest available pro model
    except Exception as e:
        st.error(f"Error initializing vision model: {str(e)}")
        st.stop()
        
    prompt_template = (  # Reusable prompt template
        "You are a professional fashion and styling assistant with expertise in creating personalized outfit recommendations.\n"
        "First, provide a section titled 'Description:' with a detailed description of the clothing item in the image.\n"
        "Next, provide a section titled 'Styling Advice:' with detailed fashion advice, including how to style and complement this item.\n"
        "Offer suggestions for pairing it with accessories, footwear, and other clothing pieces.\n"
        "Focus on the specific design elements, colors, and texture of the clothing item in the image.\n"
        "Based on the image, recommend how best to style this outfit to make a fashion statement."
    )

    if uploaded_file is not None:
        try:
            # Convert uploaded file to PIL Image directly
            uploaded_image = PIL.Image.open(uploaded_file)
            # Convert to RGB if needed (in case of RGBA images)
            if uploaded_image.mode in ('RGBA', 'LA'):
                uploaded_image = uploaded_image.convert('RGB')
            st.subheader("Uploaded Image:")
            st.image(uploaded_image, caption="Your Uploaded Image", use_container_width=True)
            
            # Convert to numpy array for ChromaDB query
            uploaded_array = np.array(uploaded_image)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            st.stop()
        
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            uploaded_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Generate response for uploaded image
            response = model.generate_content([
                prompt_template,
                {"mime_type": "image/jpeg", "data": img_str}
            ])
            response.resolve()
            
            st.subheader("Description & Styling Recommendations for Uploaded Image:")
            if hasattr(response, 'text') and response.text:
                st.write(response.text)
            else:
                st.error("No response generated from the model.")
                st.write("DEBUG RAW OUTPUT:", response)
        except Exception as e:
            st.error(f"An error occurred generating for uploaded image: {e}")
            st.write("Full error:", str(e))

        # Now retrieve and process similar images
        uploaded_array = np.array(uploaded_image)  # For query_images
        retrieved_imgs = image_vdb.query(
            query_images=[uploaded_array], 
            n_results=3, 
            include=['uris', 'distances']  # Fixed: Use 'uris' (add 'documents' if needed)
        )
        image_paths = format_image_input(retrieved_imgs)

        if image_paths:
            st.subheader("Retrieved Similar Images:")
            for i, path in enumerate(image_paths[:3]):  # Limit to 3
                try:
                    img = open_image(path)
                    st.image(img, caption=f"Retrieved Image {i+1}", use_container_width=True)
                    
                    # Convert image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    response = model.generate_content([
                        prompt_template,
                        {"mime_type": "image/jpeg", "data": img_str}
                    ])
                    response.resolve()
                    
                    st.subheader(f"Description & Styling Recommendations for Retrieved Image {i+1}:")
                    if hasattr(response, 'text') and response.text:
                        st.write(response.text)
                    else:
                        st.error("No response generated from the model.")
                        st.write("DEBUG RAW OUTPUT:", response)
                except Exception as e:
                    st.error(f"An error occurred for retrieved image {i+1}: {e}")
        else:
            st.warning("No similar images found in the database.")

    if query:
        def query_db(query_text, num_results):
            return image_vdb.query(
                query_texts=[query_text],
                n_results=num_results,
                include=['uris', 'distances']
            )

        results = query_db(query, 3)  # Increased to 3 for more options
        image_paths = format_image_input(results)

        if image_paths:
            st.subheader("Retrieved Images for Query:")
            displayed_images = []
            for i, path in enumerate(image_paths[:2]):  # Still limit display to 2 for Gemini (max ~2-3 images safely)
                try:
                    img = open_image(path)  # Fixed: Use open_image for URLs/paths
                    st.image(img, caption=f"Retrieved Image {i+1}", use_container_width=True)
                    displayed_images.append(img)
                except Exception as e:
                    st.error(f"Error loading image {i+1}: {e}")

            if len(displayed_images) >= 1:  # At least one image
                query_prompt = (
                    prompt_template.replace("the clothing item in the image.", "the clothing items in the images.") +  # Adapt for multi-image
                    f"\nThis is the piece I want to wear: {query}.\n"
                    "Based on the images, recommend how best to style these outfits to make a fashion statement."
                )
                try:
                    # Convert images to base64
                    content = [query_prompt]
                    for img in displayed_images[:2]:
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        content.append({"mime_type": "image/jpeg", "data": img_str})
                    
                    response = model.generate_content(content)
                    response.resolve()
                    
                    st.subheader("Description & Styling Recommendations:")
                    if hasattr(response, 'text') and response.text:
                        st.write(response.text)
                    else:
                        st.error("No response generated from the model.")
                        st.write("DEBUG RAW OUTPUT:", response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.write("Full error:", str(e))
            else:
                st.error("No valid images loaded for generation.")
        else:
            st.error("No images found for your query.")



        
      




    