import streamlit as st
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from PIL import Image
import io
import pandas as pd
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore

# Set page configuration
st.set_page_config(
    page_title="Solar Vision",
    page_icon="nrsc-logo-transformed.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

def welcome_page():
    st.title("Welcome to Solar Vision 🌞")
    
    st.markdown("### Solar Vision: Harnessing Automated Detection and Quantification of Solar Panels in Urban Areas to Promote Renewable Energy Adoption and Achieve Sustainable Development Goal")
    st.write("Solar Vision is an advanced pipeline that uses artificial intelligence and classical algorithms to analyze Spatial images and assess the urban envirnment for solar panel installation on rooftops or open areas with application of assesment on temporal data for analsysis.")

    st.markdown("### Key Features")
    features = [
        "AI-powered analysis of aerial images",
        "Automatic detection of suitable installation areas",
        "Calculation of Estimated power generation",
        "Estimation of installable solar panel count",
        "Generation of detailed reports"
    ]
    for feature in features:
        st.markdown(f"- {feature}")

    st.markdown("### How to Use")
    steps = [
        "This model is suggested to be used on Low Resolution Spartial Imagery.",
        "Upload a spatial image of the urban area. ",
        "Navigate to the \"Run Inference\" page using the sidebar.",
        "Upload an aerial image of the area you want to analyze",
        "Review the detailed analysis and results",
        "Download the generated report for further use"
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")

def model_matrix_page():
    st.title("Model Matrix and Training Results")

    # Read the CSV file
    try:
        df = pd.read_csv("results.csv")
        
        # Create subplots
        fig = make_subplots(rows=3, cols=2, 
                            subplot_titles=("Training Losses", "Validation Losses", 
                                            "Precision and Recall", "mAP Metrics",
                                            "Learning Rates"),
                            vertical_spacing=0.1)

        # Training losses
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train/box_loss'], name='Box Loss', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train/cls_loss'], name='Class Loss', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train/dfl_loss'], name='DFL Loss', mode='lines'), row=1, col=1)

        # Validation losses
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val/box_loss'], name='Val Box Loss', mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val/cls_loss'], name='Val Class Loss', mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val/dfl_loss'], name='Val DFL Loss', mode='lines'), row=1, col=2)

        # Precision and Recall
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/precision(B)'], name='Precision', mode='lines'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/recall(B)'], name='Recall', mode='lines'), row=2, col=1)

        # mAP Metrics
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/mAP50(B)'], name='mAP50', mode='lines'), row=2, col=2)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/mAP50-95(B)'], name='mAP50-95', mode='lines'), row=2, col=2)

        # Learning Rates
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['lr/pg0'], name='LR pg0', mode='lines'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['lr/pg1'], name='LR pg1', mode='lines'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['lr/pg2'], name='LR pg2', mode='lines'), row=3, col=1)

        # Update layout
        fig.update_layout(height=1200, title_text="Training Metrics")
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Value")

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    except FileNotFoundError:
        st.error("Error: 'results.csv' file not found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    # Add training transcript
    st.subheader("Training Transcript")
    try:
        # Attempt to read the transcript with utf-8 encoding
        with open("training_transcript.txt", "r", encoding="utf-8") as file:
            transcript = file.read()
    except FileNotFoundError:
        transcript = "Training transcript file not found."
    except UnicodeDecodeError:
        # If utf-8 fails, try ISO-8859-1 encoding
        with open("training_transcript.txt", "r", encoding="ISO-8859-1") as file:
            transcript = file.read()

    # Display the transcript in a text area
    st.text_area("Training Log", value=transcript, height=400, max_chars=None, key="transcript")

def github_page():
    st.markdown("""
    <h1 style='text-align: center;'>GitHub Repository</h1>
    <div style='background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3>Project Repository</h3>
        <p>This project is open source and available on GitHub. You can find the source code, contribute to the project, 
        or report issues through our repository.</p>
        <a href="https://github.com/ViratSrivastava/Solor-Vision-Inference" target="_blank" style="
            display: inline-block;
            padding: 10px 20px;
            background-color: #24292e;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 10px;">
            <i class="fab fa-github"></i> Visit Repository
        </a>
    </div>
    
    <div style='background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
        <h3>Features Implemented</h3>
        <ul>
            <li>YOLOV8-based solar panel detection</li>
            <li>Area calculation and power estimation</li>
            <li>Interactive web interface using Streamlit</li>
            <li>Automated report generation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def set_page_container_style():
    st.markdown("""
        <div id="starfield-container">
            <canvas id="starfield"></canvas>
        </div>
        <style>
            .reportview-container {
                background: transparent;
            }
            .main .block-container {
                background: transparent;
            }
        </style>
    """, unsafe_allow_html=True)

def load_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def set_page_container_style():
    st.markdown("""
        <div id="starfield-container">
            <canvas id="starfield"></canvas>
        </div>
        <style>
            #starfield-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
            }
            #starfield {
                width: 100%;
                height: 100%;
            }
            .reportview-container {
                background: transparent;
            }
            .main .block-container {
                background: transparent;
            }
        </style>
    """, unsafe_allow_html=True)

def mask_outside_boxes(image, labels_file):
    def read_labels(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 9:  # Assuming OBB format with 8 coordinates + class_id
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]  # Get all coordinates
                boxes.append((class_id, coords))
        
        return boxes

    def normalize_to_pixel_coordinates(box, image_width, image_height):
        return [
            (int(box[i] * image_width), int(box[i+1] * image_height))
            for i in range(0, len(box), 2)
        ]

    # Read labels and create mask
    boxes = read_labels(labels_file)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = image.shape[:2]

    # Draw the polygons on the mask
    for _, box in boxes:
        pixel_coords = normalize_to_pixel_coordinates(box, width, height)
        cv2.fillPoly(mask, [np.array(pixel_coords)], (255, 255, 255))

    # Create inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Create black background
    black_bg = np.zeros_like(image)

    # Combine original image and black background using the masks
    result = cv2.bitwise_and(image, image, mask=mask)
    result += cv2.bitwise_and(black_bg, black_bg, mask=mask_inv)

    return result, mask

def add_starfield_animation():
    with open('starfield.js', 'r') as file:
        js_code = file.read()
    
    st.markdown(f"""
    <script>
    {js_code}
    </script>
    """, unsafe_allow_html=True)

def load_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def run_inference_page():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("Solar Panel Analysis")
    st.write("Upload an image to analyze solar panels")

    # Load YOLO model
    @st.cache_resource
    def load_model():
        return YOLO('weights/best.pt')
    
    model = load_model()

    def apply_canny_edge_detection(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def count_contours(edge_image):
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        return len(contours), contour_image

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        try:
            # Create input directory if it doesn't exist
            if not os.path.exists('input'):
                os.makedirs('input')
            
            # Set fixed input path for this inference
            input_path = 'input/inference.png'
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Define a fixed prediction directory
            predict_dir = 'runs/obb/predict'
            if os.path.exists(predict_dir):
                shutil.rmtree(predict_dir)  # Remove existing prediction directory

            # Run YOLO inference
            with st.spinner("Running YOLO detection..."):
                results = model.predict(
                    source=input_path,
                    save=True,
                    save_txt=True,
                    save_json=True,
                    show_labels=True,
                    project='runs/obb',
                    name='predict'  # Use a fixed name for the prediction folder
                )

            # Update paths for the new prediction
            labels_path = f'{predict_dir}/labels/inference.txt'
            yolo_output_path = f'{predict_dir}/inference.jpg'

            # Read original and YOLO output images
            original_image = cv2.imread(input_path)
            yolo_output = cv2.imread(yolo_output_path)

            # Apply black masking
            masked_result, mask = mask_outside_boxes(original_image, labels_path)

            # Apply Canny edge detection
            edge_image = apply_canny_edge_detection(masked_result)

            # Count contours and get contour image
            num_contours, contour_image = count_contours(edge_image)

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Original & Detection", "Masked Result", "Edge Detection", "Contour Analysis"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption="Original Image")
                with col2:
                    st.image(yolo_output, caption="YOLO Detection")
            
            with tab2:
                st.image(masked_result, caption="Masked Result", channels="BGR")
            
            with tab3:
                st.image(edge_image, caption="Edge Detection Result")
            
            with tab4:
                st.image(contour_image, caption="Contour Detection")
                st.metric("Number of Detected Panels", num_contours)
                
                with st.expander("Detailed Analysis"):
                    st.write(f"""
                    - Total number of contours detected: {num_contours}
                    - Each contour represents a potential solar panel area
                    - The green outlines show the detected panel boundaries
                    """)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure the image is clear and contains solar panels.")

    st.markdown('</ div>', unsafe_allow_html=True)

def footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 1000;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
    }
    .footer a:hover {
        color: #45a049;
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <p>Developed by: 
            <a href="https://github.com/ViratSrivastava" target="_blank">Virat Srivastava GitHub</a> | 
            <a href="https://linkedin.com/in/virat-srivastava" target="_blank">Virat Srivastava LinkedIn</a> | 
            <a href="https://github.com/durgesh2411" target="_blank">Durgesh Kumar Singh GitHub</a> | 
            <a href="https://www.linkedin.com/in/durgesh-singh-745263252/" target="_blank">Durgesh Kumar Singh LinkedIn</a> | 
            <a href="https://github.com/VishuKalier2003" target="_blank">Vishu Kalier GitHub</a> | 
            <a href="https://www.linkedin.com/in/vishu-kalier-042414200/" target="_blank">Vishu Kalier LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Welcome"
    
    # Add these two lines at the beginning of main()
    set_page_container_style()
    add_starfield_animation()
    # Load custom CSS
    load_css()

    # Sidebar navigation
    st.sidebar.markdown('<h1>Navigation</h1>', unsafe_allow_html=True)
    
    # Navigation options
    pages = {
        "Welcome": welcome_page,
        "Run Inference": run_inference_page,
        "Model Matrix": model_matrix_page,
        "GitHub Repository": github_page
    }

    # Create navigation buttons
    for page, func in pages.items():
        if st.sidebar.button(page, key=page):
            st.session_state.page = page

    # Display the selected page
    if st.session_state.page in pages:
        pages[st.session_state.page]()

    # Load the image
    logo = Image.open("nrsc-logo-transformed.png")

    # Create three columns in the sidebar
    col1, col2, col3 = st.sidebar.columns([1,2,1])

    # Display the image in the middle column
    with col2:
        st.image(logo, width=150)

    # Add sidebar footer
    st.sidebar.markdown("""
        <div style="margin-top: 20px; text-align: center;">
            <hr>
            <h3>Made with ❤️ by</h3>
            <h5>
            Virat Srivastava<br>
            Durgesh Kumar Singh<br>
            Vishu Kalier
            </h5>
            Version 1.0.0
        </div>
    """, unsafe_allow_html=True)

    # Add footer
    footer()

if __name__ == "__main__":
    main()