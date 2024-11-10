import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import tempfile
import os

class WoodKnotVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.threshold_value = 30
        self.min_knot_area = 100
        
    def detect_knots(self, image):
        """Detect knots in wood using image processing."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to identify darker regions (knots)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours based on area to remove noise
        knot_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_knot_area]
        
        return knot_contours

    def find_optimal_cuts(self, image_shape, knot_contours):
        """Find optimal cutting positions to maximize clear wood sections."""
        height, width = image_shape[:2]
        knot_positions = []
        
        # Get bounding boxes for all knots
        for contour in knot_contours:
            x, y, w, h = cv2.boundingRect(contour)
            knot_positions.append((x, x + w))
        
        # Sort knot positions
        knot_positions.sort()
        
        # Find clear sections
        clear_sections = []
        last_end = 0
        
        for start, end in knot_positions:
            if start - last_end > 50:  # Minimum clear section width
                clear_sections.append((last_end, start))
            last_end = max(last_end, end)
        
        # Add final section if there's space
        if width - last_end > 50:
            clear_sections.append((last_end, width))
        
        # Find optimal cutting positions (middle of clear sections)
        cut_positions = []
        for start, end in clear_sections:
            if end - start > 100:  # Minimum section size to consider cutting
                cut_positions.append((start + end) // 2)
        
        return cut_positions, clear_sections

    def process_frame(self, img):
        """Process a single frame and return the annotated image."""
        # Detect knots
        knot_contours = self.detect_knots(img)
        
        # Draw knot contours
        cv2.drawContours(img, knot_contours, -1, (0, 0, 255), 2)
        
        # Find and draw optimal cutting positions
        cut_positions, clear_sections = self.find_optimal_cuts(img.shape, knot_contours)
        
        # Draw cutting lines
        for x in cut_positions:
            cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 2)
        
        # Draw clear sections
        for start, end in clear_sections:
            cv2.rectangle(
                img,
                (start, 0),
                (end, img.shape[0]),
                (255, 255, 0),
                1
            )
        
        # Add text overlay with measurements
        cv2.putText(
            img,
            f"Knots: {len(knot_contours)} | Cuts: {len(cut_positions)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return img

    def recv(self, frame):
        """Process each frame from the video stream."""
        img = frame.to_ndarray(format="bgr24")
        processed_img = self.process_frame(img.copy())
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def process_uploaded_video(video_file, min_knot_area):
    """Process an uploaded video file and return processed frames."""
    # Save uploaded file to temp location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Open video file
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create processor instance
    processor = WoodKnotVideoProcessor()
    processor.min_knot_area = min_knot_area
    
    # Create progress bar
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = processor.process_frame(frame)
        
        # Display frame
        frame_placeholder.image(processed_frame, channels="BGR")
        
        # Update progress
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = current_frame / frame_count
        progress_bar.progress(progress)
        
    cap.release()
    
    # Clean up temp file
    os.unlink(tfile.name)

def main():
    st.title("Wood Knot Detection")
    
    st.write("""
    This app detects knots in wood using either your webcam or an uploaded video file.
    - Red contours: Detected knots
    - Green lines: Suggested cutting positions
    - Yellow rectangles: Clear wood sections
    """)
    
    # Add configuration options
    st.sidebar.header("Detection Settings")
    min_knot_area = st.sidebar.slider(
        "Minimum Knot Area (pixels²)", 
        min_value=50,
        max_value=500,
        value=100
    )
    
    # Input selection
    input_type = st.radio("Select Input Type", ["Webcam", "Video File"])
    
    if input_type == "Webcam":
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="wood-knot-detection",
            rtc_configuration=rtc_configuration,
            video_processor_factory=WoodKnotVideoProcessor,
            async_processing=True
        )
        
        # Update processor parameters if stream is active
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.min_knot_area = min_knot_area
            
        st.markdown("""
        ### Webcam Usage Instructions:
        1. Allow camera access when prompted
        2. Position wood sample in front of camera
        3. Adjust detection settings in sidebar if needed
        4. Clear sections will be highlighted and cut lines suggested in real-time
        """)
    
    else:
        st.markdown("""
        ### Video File Instructions:
        1. Upload a video file (.mov format)
        2. Adjust detection settings in sidebar if needed
        3. The processed video will play automatically
        """)
        
        uploaded_file = st.file_uploader("Choose a video file", type=['mov'])
        
        if uploaded_file is not None:
            process_uploaded_video(uploaded_file, min_knot_area)
    
    st.markdown("""
    ### Tips:
    - Ensure good lighting on the wood surface
    - Keep the wood steady for best results
    - Try different angles to get optimal detection
    """)

if __name__ == "__main__":
    main()