import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import CropDataLoader, load_and_preprocess_data
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CropRecommendationApp:
    def __init__(self):
        self.models = {}
        self.label_encoder = None
        self.scaler = None
        self.feature_selector = None
        self.final_scaler = None
        self.data_loader = None
        self.feature_names = []
        self.target_classes = []
        self.load_data_info()
        self.load_models()
        
    def load_data_info(self):
        """Load data information dynamically"""
        try:
            self.data_loader = CropDataLoader()
            df = self.data_loader.load_data_from_file()
            self.feature_names = self.data_loader.feature_names
            self.target_classes = self.data_loader.target_classes
            print(f"Loaded data info: {len(self.feature_names)} features, {len(self.target_classes)} classes")
        except Exception as e:
            print(f"Could not load data info: {e}")
            # Fallback to default
            self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            self.target_classes = ['rice', 'maize', 'cotton', 'chickpea']
        
    def load_models(self):
        """Load trained models"""
        model_files = {
            'Random Forest': 'crop_model_random_forest.pkl',
            'Support Vector Machine': 'crop_model_support_vector_machine.pkl',
            'Logistic Regression': 'crop_model_logistic_regression.pkl',
            'Decision Tree': 'crop_model_decision_tree.pkl',
            'Gradient Boosting': 'crop_model_gradient_boosting.pkl',
            'Neural Network': 'crop_model_neural_network.pkl'
        }
        
        for name, file in model_files.items():
            try:
                with open(file, 'rb') as f:
                    self.models[name] = pickle.load(f)
            except FileNotFoundError:
                continue
        
        # Load encoders and scalers
        try:
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            st.warning("Basic model files not found.")
        
        try:
            with open('feature_selector.pkl', 'rb') as f:
                self.feature_selector = pickle.load(f)
            with open('final_scaler.pkl', 'rb') as f:
                self.final_scaler = pickle.load(f)
            print("Enhanced feature processing components loaded")
        except FileNotFoundError:
            print("Enhanced feature processing not available, using basic features")
    
    def engineer_features(self, features):
        """Apply the same feature engineering as training"""
        if len(features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")
        
        # Create feature dictionary for easier access
        feature_dict = dict(zip(self.feature_names, features))
        
        # Create enhanced features if we have the basic NPK and climate features
        enhanced_features = list(features)  # Start with original features
        
        if all(col in feature_dict for col in ['N', 'P', 'K']):
            N, P, K = feature_dict['N'], feature_dict['P'], feature_dict['K']
            # NPK ratios
            enhanced_features.extend([
                N + P + K,  # NPK_sum
                N / (P + 1e-8),  # NP_ratio
                N / (K + 1e-8),  # NK_ratio
                P / (K + 1e-8),  # PK_ratio
            ])
        
        if all(col in feature_dict for col in ['temperature', 'humidity']):
            temp, humidity = feature_dict['temperature'], feature_dict['humidity']
            enhanced_features.append(temp * humidity / 100)  # temp_humidity_index
        
        if all(col in feature_dict for col in ['rainfall', 'humidity']):
            rainfall, humidity = feature_dict['rainfall'], feature_dict['humidity']
            enhanced_features.append(rainfall * humidity / 100)  # rainfall_humidity_index
        
        if 'ph' in feature_dict:
            ph = feature_dict['ph']
            # pH categories as binary features
            enhanced_features.extend([
                1 if ph < 6.0 else 0,  # ph_acidic
                1 if 6.0 <= ph <= 7.5 else 0,  # ph_neutral
                1 if ph > 7.5 else 0,  # ph_alkaline
            ])
        
        if 'temperature' in feature_dict:
            temp = feature_dict['temperature']
            # Temperature categories
            enhanced_features.extend([
                1 if temp < 15 else 0,  # temp_cold
                1 if 15 <= temp < 25 else 0,  # temp_moderate
                1 if 25 <= temp < 35 else 0,  # temp_warm
                1 if temp >= 35 else 0,  # temp_hot
            ])
        
        if 'rainfall' in feature_dict:
            rainfall = feature_dict['rainfall']
            # Rainfall categories
            enhanced_features.extend([
                1 if rainfall < 50 else 0,  # rainfall_low
                1 if 50 <= rainfall < 100 else 0,  # rainfall_moderate
                1 if 100 <= rainfall < 200 else 0,  # rainfall_high
                1 if rainfall >= 200 else 0,  # rainfall_very_high
            ])
        
        return enhanced_features
    
    def predict_crop(self, features, model_name):
        """Make crop prediction"""
        if model_name not in self.models:
            return None, None
        
        model = self.models[model_name]
        
        try:
            if self.feature_selector and self.final_scaler:
                # Enhanced pipeline
                enhanced_features = self.engineer_features(features)
                enhanced_df = pd.DataFrame([enhanced_features])
                selected_features = self.feature_selector.transform(enhanced_df)
                features_scaled = self.final_scaler.transform(selected_features)
            else:
                # Basic pipeline
                if self.scaler:
                    features_scaled = self.scaler.transform([features])
                else:
                    features_scaled = [features]
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_crops = []
                top_probs = []
                
                if self.label_encoder:
                    for idx in top_indices:
                        crop = self.label_encoder.inverse_transform([idx])[0]
                        prob = probabilities[idx]
                        top_crops.append(crop)
                        top_probs.append(prob)
                
                return prediction, list(zip(top_crops, top_probs))
            
            if self.label_encoder:
                crop_name = self.label_encoder.inverse_transform([prediction])[0]
                return crop_name, [(crop_name, 1.0)]
            
            return prediction, [(str(prediction), 1.0)]
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Agricultural Decision Support System</p>', unsafe_allow_html=True)
    
    # Initialize app
    app = CropRecommendationApp()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    available_models = list(app.models.keys())
    if available_models:
        selected_model = st.sidebar.selectbox("Select ML Model", available_models)
    else:
        st.error("No trained models found! Please run the training scripts first.")
        st.stop()
    
    # Navigation
    page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Data Analysis", "🤖 Model Comparison", "📈 Predictions"])
    
    if page == "🏠 Home":
        show_home_page(app, selected_model)
    elif page == "📊 Data Analysis":
        show_data_analysis(app)
    elif page == "🤖 Model Comparison":
        show_model_comparison()
    elif page == "📈 Predictions":
        show_prediction_page(app, selected_model)

def show_home_page(app, selected_model):
    """Show home page with crop prediction"""
    st.markdown('<h2 class="sub-header">🌱 Get Crop Recommendations</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🧪 Soil & Climate Parameters")
        
        feature_values = {}
        feature_info = app.data_loader.get_feature_info() if app.data_loader else {}
        
        # Create input widgets dynamically
        col_a, col_b = st.columns(2)
        
        for i, feature in enumerate(app.feature_names):
            info = feature_info.get(feature, {'name': feature.title(), 'unit': '', 'description': f'{feature} parameter'})
            
            with col_a if i % 2 == 0 else col_b:
                if feature.lower() in ['n', 'nitrogen']:
                    feature_values[feature] = st.slider(f"{info['name']} ({info['unit']})", 0, 140, 50, help=info['description'])
                elif feature.lower() in ['p', 'phosphorus']:
                    feature_values[feature] = st.slider(f"{info['name']} ({info['unit']})", 5, 145, 53, help=info['description'])
                elif feature.lower() in ['k', 'potassium']:
                    feature_values[feature] = st.slider(f"{info['name']} ({info['unit']})", 5, 205, 48, help=info['description'])
                elif 'temp' in feature.lower():
                    feature_values[feature] = st.slider(f"{info['name']} ({info['unit']})", 8.8, 43.7, 25.6, help=info['description'])
                elif 'humid' in feature.lower():
                    feature_values[feature] = st.slider(f"{info['name']} ({info['unit']})", 14.3, 99.9, 71.5, help=info['description'])
                elif 'ph' in feature.lower():
                    feature_values[feature] = st.slider(f"{info['name']}", 3.5, 9.9, 6.5, 0.1, help=info['description'])
                elif 'rain' in feature.lower():
                    feature_values[feature] = st.slider(f"{info['name']} ({info['unit']})", 20.2, 298.6, 103.5, help=info['description'])
                else:
                    # Generic numeric input
                    feature_values[feature] = st.number_input(f"{info['name']} ({info['unit']})", value=0.0, help=info['description'])
        
        # Predict button
        if st.button("🔮 Get Crop Recommendation", type="primary"):
            features = [feature_values[feature] for feature in app.feature_names]
            
            prediction, top_predictions = app.predict_crop(features, selected_model)
            
            if prediction and top_predictions:
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Recommended Crop: **{top_predictions[0][0].title()}**")
                st.markdown(f"**Confidence:** {top_predictions[0][1]:.1%}")
                
                # Show top 3 predictions
                st.markdown("#### 📊 Top 3 Recommendations:")
                for i, (crop, prob) in enumerate(top_predictions[:3]):
                    st.markdown(f"{i+1}. **{crop.title()}** - {prob:.1%}")
                
                crop_info = get_dynamic_crop_info(top_predictions[0][0], app.target_classes)
                if crop_info:
                    st.markdown("#### 🌾 Growing Information:")
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.markdown(f"**Season:** {crop_info.get('season', 'N/A')}")
                        st.markdown(f"**Water Requirement:** {crop_info.get('water_req', 'N/A')}")
                    with col_info2:
                        st.markdown(f"**Soil Type:** {crop_info.get('soil_type', 'N/A')}")
                    st.markdown(f"**Tips:** {crop_info.get('tips', 'N/A')}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Input Summary")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**Model:** {selected_model}")
        for feature in app.feature_names:
            info = feature_info.get(feature, {'name': feature.title(), 'unit': ''})
            value = feature_values.get(feature, 0)
            st.markdown(f"**{info['name']}:** {value} {info['unit']}")
        st.markdown('</div>', unsafe_allow_html=True)

def show_data_analysis(app):
    """Show data analysis page"""
    st.markdown('<h2 class="sub-header">📊 Dataset Analysis</h2>', unsafe_allow_html=True)
    
    try:
        if app.data_loader:
            df = app.data_loader.load_data_from_file()
        else:
            loader = CropDataLoader()
            df = loader.load_data_from_file()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Dataset Overview")
        st.write(f"**Total Samples:** {len(df)}")
        st.write(f"**Features:** {len(df.columns)-1}")
        st.write(f"**Crops:** {df[app.data_loader.target_column].nunique()}")
        
        st.markdown("### 🔢 Statistical Summary")
        st.dataframe(df[app.feature_names].describe())
    
    with col2:
        st.markdown("### 🌾 Crop Distribution")
        crop_counts = df[app.data_loader.target_column].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        crop_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Top 10 Crops Distribution')
        ax.set_xlabel('Crops')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Feature correlations
    st.markdown("### 🔗 Feature Correlations")
    numeric_features = df[app.feature_names].select_dtypes(include=[np.number]).columns
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

def show_model_comparison():
    """Show model comparison page"""
    st.markdown('<h2 class="sub-header">🤖 Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    try:
        # Load model comparison data
        comparison_df = pd.read_csv('model_comparison.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Model Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            comparison_df.plot(x='Model', y=['Train_Accuracy', 'Test_Accuracy'], 
                             kind='bar', ax=ax, alpha=0.8)
            ax.set_title('Train vs Test Accuracy')
            ax.set_ylabel('Accuracy')
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 🏆 Model Rankings")
            sorted_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
            st.dataframe(sorted_df[['Model', 'Test_Accuracy', 'CV_Mean']])
        
        # Detailed metrics
        st.markdown("### 📈 Detailed Performance Metrics")
        st.dataframe(comparison_df)
        
    except FileNotFoundError:
        st.warning("Model comparison data not found. Please run the model training scripts first.")

def show_prediction_page(app, selected_model):
    """Show batch prediction page"""
    st.markdown('<h2 class="sub-header">📈 Batch Predictions</h2>', unsafe_allow_html=True)
    
    st.markdown("### 📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file with soil parameters", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 📊 Uploaded Data")
        st.dataframe(df.head())
        
        if st.button("🔮 Generate Predictions"):
            predictions = []
            
            for _, row in df.iterrows():
                try:
                    features = [row[col] for col in app.feature_names if col in df.columns]
                    if len(features) == len(app.feature_names):
                        pred, _ = app.predict_crop(features, selected_model)
                        predictions.append(pred)
                    else:
                        predictions.append("Invalid Data")
                except Exception as e:
                    predictions.append(f"Error: {str(e)}")
            
            df['Predicted_Crop'] = predictions
            
            st.markdown("### 🎯 Prediction Results")
            st.dataframe(df)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="crop_predictions.csv",
                mime="text/csv"
            )

def get_dynamic_crop_info(crop_name, available_crops):
    """Get dynamic crop information"""
    crop_database = {
        'rice': {
            'season': 'Kharif',
            'water_req': 'High',
            'soil_type': 'Clay loam',
            'tips': 'Requires flooded fields, warm climate, and high humidity'
        },
        'maize': {
            'season': 'Kharif/Rabi',
            'water_req': 'Medium',
            'soil_type': 'Well-drained loam',
            'tips': 'Needs warm weather and adequate rainfall during growing season'
        },
        'cotton': {
            'season': 'Kharif',
            'water_req': 'Medium',
            'soil_type': 'Black cotton soil',
            'tips': 'Requires long warm season, moderate rainfall'
        },
        'chickpea': {
            'season': 'Rabi',
            'water_req': 'Low',
            'soil_type': 'Sandy loam',
            'tips': 'Cool weather crop, drought tolerant, fixes nitrogen'
        },
        'coffee': {
            'season': 'Perennial',
            'water_req': 'High',
            'soil_type': 'Well-drained, rich soil',
            'tips': 'Requires shade, consistent rainfall, and cool temperatures'
        },
        'banana': {
            'season': 'Year-round',
            'water_req': 'High',
            'soil_type': 'Rich, well-drained',
            'tips': 'Needs warm climate, high humidity, and protection from wind'
        }
    }
    
    # Return specific info if available, otherwise generic info
    if crop_name.lower() in crop_database:
        return crop_database[crop_name.lower()]
    else:
        return {
            'season': 'Variable',
            'water_req': 'Medium',
            'soil_type': 'Well-drained',
            'tips': f'Follow local agricultural guidelines for optimal {crop_name} cultivation'
        }

if __name__ == "__main__":
    main()
