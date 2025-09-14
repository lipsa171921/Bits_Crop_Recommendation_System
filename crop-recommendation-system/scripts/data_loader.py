import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import os
import requests
import io

class CropDataLoader:
    def __init__(self, csv_file="Crop_recommendation.csv"):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.csv_file = csv_file
        self.feature_names = []
        self.target_column = None
        self.target_classes = []

    def load_data_from_file(self):
        """Load dataset from file"""
        try:
            # Read CSV file
            df = pd.read_csv(self.csv_file)

            print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            # Auto-detect target column (assume it's the last column or named 'label')
            if 'label' in df.columns:
                self.target_column = 'label'
            else:
                self.target_column = df.columns[-1]  # Last column as target

            # Auto-detect feature columns (all except target)
            self.feature_names = [col for col in df.columns if col != self.target_column]

            # Auto-detect target classes
            self.target_classes = sorted(df[self.target_column].unique())

            print(f"Target column: {self.target_column}")
            print(f"Feature columns: {self.feature_names}")
            print(f"Target classes ({len(self.target_classes)}): {self.target_classes}")

            for col in self.feature_names:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"Converted {col} to numeric")
                    except:
                        print(f"Could not convert {col} to numeric - keeping as categorical")

            return df

        except Exception as e:
            error_msg = f"Error processing dataset: {e}\nPlease ensure the CSV file is valid."
            raise RuntimeError(error_msg)

    def load_data_from_other_folders(self, csv_filename="Crop_recommendation.csv"):
        """Load dataset from other folders if available"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(script_dir, csv_filename),
                os.path.join(script_dir, "..", csv_filename),
                os.path.join(script_dir, "data", csv_filename),
                csv_filename  # Current directory
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path is None:
                raise FileNotFoundError(
                    f"CSV file '{csv_filename}' not found in any of these locations:\n" +
                    "\n".join([f"  - {path}" for path in possible_paths]) +
                    f"\n\nPlease place the '{csv_filename}' file in the scripts folder."
                )
            
            print(f"Loading dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            print(f"Dataset loaded successfully! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Auto-detect target column (assume it's the last column or named 'label')
            if 'label' in df.columns:
                self.target_column = 'label'
            else:
                self.target_column = df.columns[-1]  # Last column as target
            
            # Auto-detect feature columns (all except target)
            self.feature_names = [col for col in df.columns if col != self.target_column]
            
            # Auto-detect target classes
            self.target_classes = sorted(df[self.target_column].unique())
            
            print(f"Target column: {self.target_column}")
            print(f"Feature columns: {self.feature_names}")
            print(f"Target classes ({len(self.target_classes)}): {self.target_classes}")
            
            for col in self.feature_names:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"✅ Converted {col} to numeric")
                    except:
                        print(f"⚠️ Could not convert {col} to numeric - keeping as categorical")
            
            return df
            
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            error_msg = f"Error processing dataset: {e}\nPlease ensure the CSV file is valid."
            raise RuntimeError(error_msg)
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        missing_cols = [col for col in self.feature_names + [self.target_column] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_clean = df.dropna()
        print(f"Removed {len(df) - len(df_clean)} rows with missing values")
        
        X = df_clean[self.feature_names].copy()
        y = df_clean[self.target_column].copy()
        
        categorical_features = X.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            print(f"Found categorical features: {categorical_features.tolist()}")
            # For now, we'll encode categorical features as numeric
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            X_scaled = X.copy()
            X_scaled[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        else:
            X_scaled = X.copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test, X_scaled, y_encoded
    
    def get_crop_info(self):
        """Get dynamic crop information based on detected classes"""
        crop_info = {}
        
        # Default info template
        default_info = {
            'season': 'Variable',
            'water_req': 'Medium',
            'soil_type': 'Well-drained',
            'tips': 'Follow local agricultural guidelines for optimal growth'
        }
        
        # Create info for each detected class
        for crop in self.target_classes:
            crop_info[crop.lower()] = default_info.copy()
            crop_info[crop.lower()]['tips'] = f'Optimal growing conditions for {crop} based on soil and climate parameters'
        
        return crop_info
    
    def get_feature_info(self):
        """Get dynamic feature information"""
        feature_info = {}
        
        for feature in self.feature_names:
            if feature.lower() in ['n', 'nitrogen']:
                feature_info[feature] = {
                    'name': 'Nitrogen',
                    'unit': 'kg/ha',
                    'description': 'Nitrogen content in soil'
                }
            elif feature.lower() in ['p', 'phosphorus']:
                feature_info[feature] = {
                    'name': 'Phosphorus', 
                    'unit': 'kg/ha',
                    'description': 'Phosphorus content in soil'
                }
            elif feature.lower() in ['k', 'potassium']:
                feature_info[feature] = {
                    'name': 'Potassium',
                    'unit': 'kg/ha', 
                    'description': 'Potassium content in soil'
                }
            elif 'temp' in feature.lower():
                feature_info[feature] = {
                    'name': 'Temperature',
                    'unit': '°C',
                    'description': 'Average temperature'
                }
            elif 'humid' in feature.lower():
                feature_info[feature] = {
                    'name': 'Humidity',
                    'unit': '%',
                    'description': 'Relative humidity'
                }
            elif 'ph' in feature.lower():
                feature_info[feature] = {
                    'name': 'pH Level',
                    'unit': '',
                    'description': 'Soil pH level'
                }
            elif 'rain' in feature.lower():
                feature_info[feature] = {
                    'name': 'Rainfall',
                    'unit': 'mm',
                    'description': 'Annual rainfall'
                }
            else:
                feature_info[feature] = {
                    'name': feature.title(),
                    'unit': '',
                    'description': f'{feature.title()} parameter'
                }
        
        return feature_info

@st.cache_data
def load_and_preprocess_data(csv_file="Crop_recommendation.csv"):
    """Load and preprocess data with caching for Streamlit"""
    loader = CropDataLoader(csv_file)

    try:
        df = loader.load_data_from_file()
    except Exception as e:
        print(f"Failed to load from file: {e}")
        print("Trying to load from local folders...")
        df = loader.load_data_from_other_folders()
    
    X_train, X_test, y_train, y_test, X_scaled, y_encoded = loader.preprocess_data(df)
    return loader, df, X_train, X_test, y_train, y_test, X_scaled, y_encoded
