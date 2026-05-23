"""
Streamlit App for Glass Identification using ANN

This app provides a user-friendly interface for real-time glass identification
using a trained Artificial Neural Network model.
Features:
- Interactive input fields for glass features
- Real-time predictions with confidence scores
- Detailed dataset information and feature descriptions
- Responsive design with custom styling
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import torch
import torch.nn as nn

# Page configuration
st.set_page_config(
    page_title="ANN Glass Identifier",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS (Updated to Ice Blue Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4682B4; /* Changed from red to Steel/Ice Blue */
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2A4D69; /* Darker ice blue accent for subheadings */
        margin-top: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .success {
        background-color: #E0F7FA; /* Light ice blue background */
        border: 2px solid #00BCD4; /* Soft cyan/ice blue border */
    }
    .info-box {
        /* Uses the theme's native secondary background color */
        background-color: var(--secondary-background-color); 
        /* Uses the theme's native body text color for a clean contrast border */
        border: 1px solid var(--text-color); 
        opacity: 0.85;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_artifacts():
    """
    Load the trained model and preprocessing artifacts.
    
    Returns:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        target_names: List of target class names
    """
    try:
        # Load model architecture info
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        # Define the model architecture
        class GlassANN(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(GlassANN, self).__init__()
                self.layer1 = nn.Linear(input_dim, 128)
                self.dropout1 = nn.Dropout(0.2)
                self.layer2 = nn.Linear(128, 64)
                self.dropout2 = nn.Dropout(0.2)
                self.layer3 = nn.Linear(64, num_classes)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.dropout1(x)
                x = self.relu(self.layer2(x))
                x = self.dropout2(x)
                x = self.layer3(x)
                return x
        
        # Load the model state
        model = GlassANN(model_info['input_dim'], model_info['num_classes'])
        model.load_state_dict(torch.load('glass_ann_model.pth', map_location='cpu'))
        model.eval()
        
        # Load scaler and metadata
        scaler = joblib.load('scaler.pkl')
        
        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, metadata['feature_names'], metadata['target_names']
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run train_model.py first.")
        st.stop()


def get_feature_ranges():
    """
    Return typical ranges for glass features to guide user input.
    
    Returns:
        dict: Feature ranges and descriptions
    """
    return { #1	1.52101	13.64	4.49	1.10	71.78	0.06	8.75	0.0	0.0	1
        'RI': { # Refractive Index - (1.51115, 1.53393)
            'min': 1.51115, 'max': 1.53393, 'default': 1.52101,
            'description': 'Refractive Index'
        },
        'Na': { # Sodium content - (10.73, 17.38)
            'min': 10.73, 'max': 17.38, 'default': 13.64,
            'description': 'Sodium content (wt%)'
        },
        'Mg': { # Magnesium content - (0.0, 4.49)
            'min': 0.0, 'max': 4.49, 'default': 4.49,
            'description': 'Magnesium content (wt%)'
        },
        'Al': { # Aluminum content - (0.0, 3.5)
            'min': 0.0, 'max': 3.5, 'default': 1.10,
            'description': 'Aluminum content (wt%)'
        },
        'Si': { # Silicon content - (69.0, 78.0)
            'min': 69.0, 'max': 78.0, 'default': 71.78,
            'description': 'Silicon content (wt%)'
        },
        'K': { # Potassium content - (0.0, 6.2)
            'min': 0.0, 'max': 6.2, 'default': 0.06,
            'description': 'Potassium content (wt%)'
        },
        'Ca': { # Calcium content - (5.0, 18.0)
            'min': 5.0, 'max': 18.0, 'default': 8.75,
            'description': 'Calcium content (wt%)'
        },
        'Ba': { # Barium content - (0.0, 3.1)
            'min': 0.0, 'max': 3.1, 'default': 0.0,
            'description': 'Barium content (wt%)'
        },
        'Fe': { # Iron content - (0.0, 1.2)
            'min': 0.0, 'max': 1.2, 'default': 0.0,
            'description': 'Iron content (wt%)'
        }
    }


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🧊 ANN Glass Identifier</div>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">'
                'Identify glass types using a trained Artificial Neural Network</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model and artifacts
    model, scaler, feature_names, target_names = load_model_and_artifacts()
    feature_names = [f for f in feature_names if f != 'Type']
    
    feature_ranges = get_feature_ranges()
    
    # Sidebar for input
    st.sidebar.markdown('<div class="sub-header">📊 Glass Features</div>', 
                       unsafe_allow_html=True)
    st.sidebar.info("Enter the chemical properties of the glass sample below:")
    
    # Create input fields for each feature
    input_data = {}
    
    for feature in feature_names:
        range_info = feature_ranges.get(feature, {})
        default = range_info.get('default', 0)
        min_val = range_info.get('min', 0)
        max_val = range_info.get('max', 100)
        desc = range_info.get('description', '')
        
        if desc:
            st.sidebar.caption(f"{desc}")
        
        input_data[feature] = st.sidebar.number_input(
            label=feature.replace('_', ' ').title(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.01,
            format="%.2f",
            key=feature
        )
    
    # Sidebar buttons
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    with col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    # Reset button functionality
    if reset_btn:
        st.rerun()
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="sub-header">📝 Input Summary</div>', 
                   unsafe_allow_html=True)
        
        # Display input data in a table
# Display input data in a table
        input_df = pd.DataFrame([input_data])
        
        # --- CHANGE START: Clean up row names ---
        # If "Type" is still sneaking in, drop it first
        if 'Type' in input_df.columns:
            input_df = input_df.drop(columns=['Type'])
            
        # Rename columns so they look clean when flipped into rows
        input_df = input_df.rename(columns={
            'RI': 'Refractive Index (RI)',
            'Na': 'Sodium (Na)',
            'Mg': 'Magnesium (Mg)',
            'Al': 'Aluminum (Al)',
            'Si': 'Silicon (Si)',
            'K':  'Potassium (K)',
            'Ca': 'Calcium (Ca)',
            'Ba': 'Barium (Ba)',
            'Fe': 'Iron (Fe)'
        })
        # --- CHANGE END ---
        
        st.dataframe(input_df.T, use_container_width=True, 
                    column_config={0: "Weight %", 1: "Value"},
                    height=352)
    
    with col_right:
        st.markdown('<div class="sub-header">ℹ️ Dataset Info</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Glass Dataset</strong><br><br>
        - <strong>Classes:</strong> 7 (Glass Types)<br>
        - <strong>Features:</strong> 9 chemical properties<br>
        - <strong>Samples:</strong> 214<br><br>
        <strong>Classes:</strong><br>
        - Class 1: Building Windows (Float Processed)<br>
        - Class 2: Building Windows (Non-Float Processed)<br>
        - Class 3: Vehicle Windows (Float Processed)<br>
        - Class 4: Vehicle Windows (Non-Float Processed)<br>
        - Class 5: Containers<br>
        - Class 6: Tableware<br>
        - Class 7: Headlamps
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction
    if predict_btn:
        st.markdown("---")
        st.markdown('<div class="sub-header">🎯 Prediction Results</div>', 
                   unsafe_allow_html=True)
        
        # Prepare input for prediction
        input_array = np.array([list(input_data.values())])

        # Scale the inputs completely
        input_scaled = scaler.transform(input_array)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        with st.spinner("Making prediction..."):
            # Convert to PyTorch tensor
            input_tensor = torch.FloatTensor(input_scaled)
            with torch.no_grad():
                outputs = model(input_tensor)
                prediction_probs = torch.softmax(outputs, dim=1).numpy()
                predicted_class = np.argmax(prediction_probs, axis=1)[0]
                confidence = prediction_probs[0][predicted_class] * 100
        
        # Display results
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            predicted_class_name = target_names[predicted_class]
            st.markdown(f"""
            <div class="result-box success">
                <h2 style="margin: 0; color: #006064;">
                    🧊 Predicted Class: {predicted_class_name.replace('_', ' ').title()}
                </h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; color: #00838F; font-weight: 500;">
                    🔍 We have solved the mystery of what type of glass! 
                </p>
                <p style="margin: 0.2rem 0 0 0; font-size: 1.1rem; color: #B71C1C; font-weight: bold;">
                    🚨 Case Closed: We're getting the killer!
                </p>
                <hr style="margin: 0.5rem 0; border-color: rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 1rem; color: #555;">
                    Confidence Threshold: <strong>{confidence:.2f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Class probabilities
            st.markdown("### 📊 Class Probabilities")
            
            prob_df = pd.DataFrame({
                'Class': target_names,
                'Probability': [f"{prob*100:.2f}%" for prob in prediction_probs[0]]
            })
            
            # Highlight the predicted class
            def highlight_max(s):
                is_max = s.index == predicted_class
                return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_max]
            
            st.dataframe(prob_df, use_container_width=True)
        
        with result_col2:
            # Confidence meter
            st.markdown("### 🔬 Confidence")
            
            # Confidence display (using metric instead of progress)
            st.metric("Confidence Score", f"{confidence:.1f}%")
            
            # Interpretation
            if confidence >= 90:
                st.success("✅ Very confident prediction")
            elif confidence >= 70:
                st.info("ℹ️ Confident prediction")
            elif confidence >= 50:
                st.warning("⚠️ Moderate confidence")
            else:
                st.error("❌ Low confidence - consider reviewing input values")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9rem;">
        <p>Built with ❤️ using Streamlit, TensorFlow, and scikit-learn</p>
        <p>Model: Artificial Neural Network (2 hidden layers)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()