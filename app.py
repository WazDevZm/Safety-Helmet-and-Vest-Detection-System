"""
Safety Helmet and Vest Detection System - Streamlit Dashboard
Main application for PPE detection with image upload functionality
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from ppe_detector import PPEDetector
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Safety Helmet and Vest Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .safety-alert {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .safety-ok {
        background-color: #44ff44;
        color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load the PPE detector model (cached for performance)"""
    return PPEDetector()

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🦺 Safety Helmet and Vest Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: #666;">
            Upload an image of workers to detect safety helmets and reflective vests.<br>
            The system will identify missing PPE and provide safety compliance analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        # Detection confidence threshold
        confidence_threshold = st.slider(
            "Detection Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Minimum confidence for person detection"
        )
        
        # Alert settings
        st.header("🚨 Alert Settings")
        enable_alerts = st.checkbox("Enable Safety Alerts", value=True)
        alert_sound = st.checkbox("Enable Alert Sound", value=False)
        
        # Display settings
        st.header("📊 Display Options")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_statistics = st.checkbox("Show Detailed Statistics", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing workers to analyze for PPE compliance"
        )
        
        if uploaded_file is not None:
            # Load and display the uploaded image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert PIL to OpenCV format
            if len(image_array.shape) == 3:
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_array
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Analyze PPE Compliance", type="primary"):
                with st.spinner("Analyzing image for PPE compliance..."):
                    # Load detector
                    detector = load_detector()
                    
                    # Run detection
                    start_time = time.time()
                    results = detector.detect_ppe(image_cv)
                    processing_time = time.time() - start_time
                    
                    # Draw detections on image
                    result_image = detector.draw_detections(image_cv, results['detections'])
                    
                    # Convert back to RGB for display
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    # Display results
                    st.header("🎯 Detection Results")
                    st.image(result_image_rgb, caption="PPE Detection Results", use_column_width=True)
                    
                    # Processing time
                    st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
    
    with col2:
        st.header("📊 Safety Status")
        
        if uploaded_file is not None and 'results' in locals():
            # Safety compliance status
            compliance = results['ppe_compliance']
            
            if compliance['compliance_rate'] == 100:
                st.markdown('<div class="safety-ok">✅ ALL WORKERS ARE SAFE</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="safety-alert">⚠️ SAFETY VIOLATIONS DETECTED</div>', 
                           unsafe_allow_html=True)
            
            # Key metrics
            st.metric("Compliance Rate", f"{compliance['compliance_rate']:.1f}%")
            st.metric("Total Workers", compliance['total_persons'])
            st.metric("Compliant Workers", compliance['compliant_persons'])
            
            if compliance['missing_helmets'] > 0:
                st.metric("Missing Helmets", compliance['missing_helmets'], delta="⚠️")
            if compliance['missing_vests'] > 0:
                st.metric("Missing Vests", compliance['missing_vests'], delta="⚠️")
            
            # Detailed statistics
            if show_statistics:
                st.header("📈 Detailed Analysis")
                
                # Create compliance chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Compliance pie chart
                labels = ['Compliant', 'Non-Compliant']
                sizes = [compliance['compliant_persons'], 
                        compliance['total_persons'] - compliance['compliant_persons']]
                colors = ['#2ecc71', '#e74c3c']
                
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('PPE Compliance Distribution')
                
                # PPE items chart
                ppe_items = ['Helmets', 'Vests']
                missing_counts = [compliance['missing_helmets'], compliance['missing_vests']]
                
                ax2.bar(ppe_items, missing_counts, color=['#f39c12', '#9b59b6'])
                ax2.set_title('Missing PPE Items')
                ax2.set_ylabel('Count')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Individual worker analysis
            if results['detections']:
                st.header("👷 Individual Worker Analysis")
                
                for i, detection in enumerate(results['detections']):
                    with st.expander(f"Worker {i+1} - {'✅ Safe' if detection['ppe_compliant'] else '⚠️ Unsafe'}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Helmet:** {'✅ Detected' if detection['helmet_detected'] else '❌ Missing'}")
                            st.write(f"**Vest:** {'✅ Detected' if detection['vest_detected'] else '❌ Missing'}")
                        
                        with col_b:
                            if show_confidence:
                                st.write(f"**Confidence:** {detection['confidence']:.2f}")
                            
                            if detection['missing_ppe']:
                                st.write(f"**Missing PPE:** {', '.join(detection['missing_ppe'])}")
        
        else:
            st.info("👆 Upload an image to begin PPE analysis")
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions:
    1. **Upload Image**: Click "Browse files" to select an image containing workers
    2. **Analyze**: Click "Analyze PPE Compliance" to run the detection
    3. **Review Results**: Check the safety status and detailed analysis
    4. **Take Action**: Address any missing PPE violations immediately
    
    ### 🎯 What the System Detects:
    - **Safety Helmets**: Hard hats and protective headgear
    - **Reflective Vests**: High-visibility safety vests
    - **Compliance Status**: Overall safety compliance for each worker
    
    ### ⚠️ Safety Reminders:
    - Always ensure workers wear proper PPE in hazardous areas
    - Regular PPE inspections are essential for workplace safety
    - This system is a tool to assist with safety compliance monitoring
    """)

if __name__ == "__main__":
    main()
