import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import os
import requests
import io

class CropDataLoader:
    def __init__(self, 
                 train_csv="train.csv",
                 test_csv="test.csv"):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.feature_names = []  # Will be auto-detected
        self.target_column = None  # Will be auto-detected
        self.target_classes = []  # Will be auto-detected
        
    def load_data_from_file(self, use_test_data=False):
        """Load dataset from file - supports both train and test datasets"""
        data_file = self.test_csv if use_test_data else self.train_csv
        dataset_type = "test" if use_test_data else "train"

        try:
            print(f"Fetching {dataset_type} dataset from file...")

            # Read CSV file
            df = pd.read_csv(data_file)

            print(f"Dataset loaded successfully! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            print(f"{dataset_type.title()} dataset loaded successfully! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Auto-detect target column (assume it's the first column named 'label')

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
            error_msg = f"Error processing {dataset_type} dataset: {e}\nPlease ensure the CSV file is valid."
            raise RuntimeError(error_msg)
    
    def load_both_datasets(self):
        """Load both train and test datasets"""
        try:
            print("📥 Loading both train and test datasets...")
            train_df = self.load_data_from_file(use_test_data=False)
            test_df = self.load_data_from_file(use_test_data=True)
            
            # Combine for overall statistics but keep separate for proper ML evaluation
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            
            print(f"Combined dataset shape: {combined_df.shape}")
            print(f"Train: {train_df.shape}, Test: {test_df.shape}")
            
            return train_df, test_df, combined_df
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            print("Falling back to single dataset approach...")
            # Fallback to single dataset
            df = self.load_data_from_file(use_test_data=False)
            return df, None, df
    
    def preprocess_data(self, df, test_df=None):
        """Preprocess the dataset - supports separate train/test datasets"""

        missing_cols = [col for col in self.feature_names + [self.target_column] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_clean = df.dropna()
        print(f"Removed {len(df) - len(df_clean)} rows with missing values from training data")
        
        X_train = df_clean[self.feature_names].copy()
        y_train = df_clean[self.target_column].copy()
        
        # Handle test data if provided separately
        if test_df is not None:
            test_df_clean = test_df.dropna()
            print(f"Removed {len(test_df) - len(test_df_clean)} rows with missing values from test data")
            X_test = test_df_clean[self.feature_names].copy()
            y_test = test_df_clean[self.target_column].copy()
        else:
            # Split training data if no separate test set
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Handle categorical features
        categorical_features = X_train.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            print(f"Found categorical features: {categorical_features.tolist()}")
            for col in categorical_features:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
        
        # Encode target labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Scale numeric features
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            X_train_scaled[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
            X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
        else:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
        
        # Combine scaled data for compatibility
        X_scaled = pd.concat([X_train_scaled, X_test_scaled], ignore_index=True)
        y_encoded = np.concatenate([y_train_encoded, y_test_encoded])
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, X_scaled, y_encoded
    
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

