"""
Fixed Safety Helmet and Vest Detection System - Streamlit App
Fixed version that properly flags missing PPE
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from ppe_detector_fixed import PPEDetectorFixed

# Configure page
st.set_page_config(
    page_title="Fixed Safety Helmet and Vest Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
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
        border: 3px solid #ff0000;
        animation: pulse 2s infinite;
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
        border: 3px solid #00ff00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ppe-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-weight: bold;
    }
    .ppe-detected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .ppe-missing {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load the fixed PPE detector model (cached for performance)"""
    return PPEDetectorFixed()

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🦺 Fixed Safety Helmet and Vest Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: #666;">
            <strong>Enhanced PPE Detection with Proper Flagging</strong><br>
            Upload an image of workers to detect safety helmets and reflective vests.<br>
            The system will properly identify missing PPE and provide detailed safety compliance analysis.
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
            value=0.3, 
            step=0.1,
            help="Minimum confidence for person detection (lower = more sensitive)"
        )
        
        # Alert settings
        st.header("🚨 Alert Settings")
        enable_alerts = st.checkbox("Enable Safety Alerts", value=True)
        show_missing_details = st.checkbox("Show Missing PPE Details", value=True)
        
        # Display settings
        st.header("📊 Display Options")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_statistics = st.checkbox("Show Detailed Statistics", value=True)
        show_individual_analysis = st.checkbox("Show Individual Worker Analysis", value=True)
    
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
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("🔍 Analyze PPE Compliance", type="primary"):
                with st.spinner("Analyzing image for PPE compliance with fixed detection..."):
                    try:
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
                        st.header("🎯 Fixed Detection Results")
                        st.image(result_image_rgb, caption="PPE Detection Results with Proper Flagging", use_container_width=True)
                        
                        # Processing time
                        st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
                        
                        # Store results in session state for sidebar display
                        st.session_state['results'] = results
                        st.session_state['processing_time'] = processing_time
                        
                    except Exception as e:
                        st.error(f"❌ Error during detection: {str(e)}")
                        st.info("💡 Try uploading a different image or check the image quality")
    
    with col2:
        st.header("📊 Safety Status")
        
        if 'results' in st.session_state:
            # Safety compliance status
            compliance = st.session_state['results']['ppe_compliance']
            
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
                compliance_data = {
                    'Compliant': compliance['compliant_persons'],
                    'Non-Compliant': compliance['total_persons'] - compliance['compliant_persons']
                }
                
                st.bar_chart(compliance_data)
                
                # PPE items chart
                ppe_data = {
                    'Missing Helmets': compliance['missing_helmets'],
                    'Missing Vests': compliance['missing_vests']
                }
                
                st.bar_chart(ppe_data)
            
            # Individual worker analysis
            if show_individual_analysis and st.session_state['results']['detections']:
                st.header("👷 Individual Worker Analysis")
                
                for i, detection in enumerate(st.session_state['results']['detections']):
                    # Determine status color
                    status_class = "ppe-detected" if detection['ppe_compliant'] else "ppe-missing"
                    status_icon = "✅" if detection['ppe_compliant'] else "❌"
                    status_text = "SAFE" if detection['ppe_compliant'] else "UNSAFE"
                    
                    with st.expander(f"{status_icon} Worker {i+1} - {status_text}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # Helmet status
                            helmet_class = "ppe-detected" if detection['helmet_detected'] else "ppe-missing"
                            helmet_icon = "✅" if detection['helmet_detected'] else "❌"
                            helmet_text = "Detected" if detection['helmet_detected'] else "Missing"
                            
                            st.markdown(f"""
                            <div class="ppe-status {helmet_class}">
                                {helmet_icon} <strong>Helmet:</strong> {helmet_text}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Vest status
                            vest_class = "ppe-detected" if detection['vest_detected'] else "ppe-missing"
                            vest_icon = "✅" if detection['vest_detected'] else "❌"
                            vest_text = "Detected" if detection['vest_detected'] else "Missing"
                            
                            st.markdown(f"""
                            <div class="ppe-status {vest_class}">
                                {vest_icon} <strong>Vest:</strong> {vest_text}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            if show_confidence:
                                st.write(f"**Confidence:** {detection['confidence']:.2f}")
                            
                            if detection['missing_ppe']:
                                st.write(f"**Missing PPE:** {', '.join(detection['missing_ppe'])}")
                            
                            # Safety status
                            safety_class = "ppe-detected" if detection['ppe_compliant'] else "ppe-missing"
                            st.markdown(f"""
                            <div class="ppe-status {safety_class}">
                                <strong>Safety Status:</strong> {detection['safety_status']}
                            </div>
                            """, unsafe_allow_html=True)
        
        else:
            st.info("👆 Upload an image to begin PPE analysis")
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Fixed Detection Features:
    1. **Proper PPE Flagging**: Correctly identifies missing helmets and vests
    2. **Strict Detection Criteria**: Only flags obvious PPE violations
    3. **Visual Safety Alerts**: Clear color-coded status indicators
    4. **Detailed Analysis**: Individual worker PPE status and missing equipment
    5. **Real-time Feedback**: Immediate safety compliance assessment
    
    ### 🎯 What the Fixed System Detects:
    - **Safety Helmets**: Only very bright, obvious helmets
    - **Reflective Vests**: Only very bright, high-visibility vests
    - **Missing PPE**: Properly flags workers without equipment
    - **Safety Violations**: Clear alerts for non-compliance
    
    ### ⚠️ Safety Reminders:
    - Always ensure workers wear proper PPE in hazardous areas
    - Regular PPE inspections are essential for workplace safety
    - This fixed system provides accurate detection and proper flagging of safety violations
    """)

if __name__ == "__main__":
    main()
