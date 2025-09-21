import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import io
import pickle
from data_loader import CropDataLoader

def load_crop_data():
    """Load the crop recommendation dataset using CropDataLoader"""
    try:
        loader = CropDataLoader()
        
        # Try URL first, then fallback to local file
        try:
            df = loader.load_data_from_file()
        except Exception as e:
            print(f"Failed to load from file: {e}")
            print("Trying to load from local file...")
            df = loader.load_data_from_file()
        
        return df, loader
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None

def explore_data(df, loader):
    """Perform exploratory data analysis"""
    print("\n=== DATASET OVERVIEW ===")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nDataset Description:")
    print(df.describe())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    print(f"\nCrop Distribution:")
    print(df[loader.target_column].value_counts())
    
    print(f"\nFeature Correlations:")
    numeric_features = [col for col in loader.feature_names if df[col].dtype in ['int64', 'float64']]
    correlation_matrix = df[numeric_features].corr()
    print(correlation_matrix)
    
    return correlation_matrix

def visualize_data(df, loader):
    """Create visualizations for the dataset"""
    plt.style.use('default')
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Crop Recommendation Dataset Analysis', fontsize=16, fontweight='bold')
    
    crop_counts = df[loader.target_column].value_counts().head(10)
    axes[0, 0].bar(range(len(crop_counts)), crop_counts.values, color='skyblue')
    axes[0, 0].set_title('Top 10 Crop Distribution')
    axes[0, 0].set_xlabel('Crops')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(range(len(crop_counts)))
    axes[0, 0].set_xticklabels(crop_counts.index, rotation=45)
    
    numeric_features = [col for col in loader.feature_names if df[col].dtype in ['int64', 'float64']]
    
    # Temperature vs Humidity (if available)
    temp_col = next((col for col in numeric_features if 'temp' in col.lower()), None)
    humid_col = next((col for col in numeric_features if 'humid' in col.lower()), None)
    
    if temp_col and humid_col:
        crops_sample = df[loader.target_column].unique()[:5]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, crop in enumerate(crops_sample):
            crop_data = df[df[loader.target_column] == crop]
            axes[0, 1].scatter(crop_data[temp_col], crop_data[humid_col], 
                              alpha=0.6, label=crop, color=colors[i])
        axes[0, 1].set_title(f'{temp_col.title()} vs {humid_col.title()} by Crop')
        axes[0, 1].set_xlabel(f'{temp_col.title()}')
        axes[0, 1].set_ylabel(f'{humid_col.title()}')
        axes[0, 1].legend()
    else:
        # Plot first two numeric features if temp/humidity not available
        if len(numeric_features) >= 2:
            axes[0, 1].scatter(df[numeric_features[0]], df[numeric_features[1]], alpha=0.5)
            axes[0, 1].set_title(f'{numeric_features[0]} vs {numeric_features[1]}')
            axes[0, 1].set_xlabel(numeric_features[0])
            axes[0, 1].set_ylabel(numeric_features[1])
    
    # NPK nutrients distribution (if available)
    npk_cols = [col for col in numeric_features if col.upper() in ['N', 'P', 'K']]
    if npk_cols:
        axes[1, 0].boxplot([df[col] for col in npk_cols], labels=npk_cols)
        axes[1, 0].set_title('NPK Nutrients Distribution')
        axes[1, 0].set_ylabel('Concentration')
    else:
        # Plot distribution of first 3 numeric features
        plot_cols = numeric_features[:3]
        if plot_cols:
            axes[1, 0].boxplot([df[col] for col in plot_cols], labels=plot_cols)
            axes[1, 0].set_title('Feature Distribution')
            axes[1, 0].set_ylabel('Values')
    
    # pH vs Rainfall (if available)
    ph_col = next((col for col in numeric_features if 'ph' in col.lower()), None)
    rain_col = next((col for col in numeric_features if 'rain' in col.lower()), None)
    
    if ph_col and rain_col:
        axes[1, 1].scatter(df[ph_col], df[rain_col], alpha=0.5, color='green')
        axes[1, 1].set_title(f'{ph_col.title()} vs {rain_col.title()}')
        axes[1, 1].set_xlabel(ph_col.title())
        axes[1, 1].set_ylabel(rain_col.title())
    else:
        # Plot last two numeric features
        if len(numeric_features) >= 2:
            axes[1, 1].scatter(df[numeric_features[-2]], df[numeric_features[-1]], alpha=0.5, color='green')
            axes[1, 1].set_title(f'{numeric_features[-2]} vs {numeric_features[-1]}')
            axes[1, 1].set_xlabel(numeric_features[-2])
            axes[1, 1].set_ylabel(numeric_features[-1])
    
    plt.tight_layout()
    plt.savefig('crop_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    if len(numeric_features) > 1:
        correlation_matrix = df[numeric_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

def preprocess_data(df, loader):
    """Preprocess the data for machine learning using CropDataLoader"""
    print("\n=== DATA PREPROCESSING ===")
    
    X_train, X_test, y_train, y_test, X_scaled, y_encoded = loader.preprocess_data(df)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of unique crops: {len(loader.target_classes)}")
    print(f"Crop classes: {loader.target_classes}")
    
    return X_train, X_test, y_train, y_test, loader.scaler, loader.label_encoder, loader.feature_names

def main():
    """Main function to run data analysis"""
    print("Starting Crop Recommendation Data Analysis...")
    
    df, loader = load_crop_data()
    if df is None or loader is None:
        return
    
    # Explore data
    correlation_matrix = explore_data(df, loader)
    
    # Visualize data
    visualize_data(df, loader)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols = preprocess_data(df, loader)
    
    # Save preprocessed data and encoders for model training
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save encoders and feature names
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("\nData analysis completed successfully!")
    print("Preprocessed data and encoders saved for model training.")

if __name__ == "__main__":
    main()
