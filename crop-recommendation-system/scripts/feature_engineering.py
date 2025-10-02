import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import pickle


def drop_constant_columns(df):
    """Remove columns with constant values"""
    return df.loc[:, df.nunique() > 1]


def create_feature_combinations(df):
    """Create new features from existing ones"""
    print("Creating feature combinations...")
    
    # Create NPK ratio features
    df['NPK_sum'] = df['N'] + df['P'] + df['K']
    df['NP_ratio'] = df['N'] / (df['P'] + 1e-8)  # Add small value to avoid division by zero
    df['NK_ratio'] = df['N'] / (df['K'] + 1e-8)
    df['PK_ratio'] = df['P'] / (df['K'] + 1e-8)

    # Create climate comfort indices
    # df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
    # df['rainfall_humidity_index'] = df['rainfall'] * df['humidity'] / 100
    
    # Create soil quality indicators
    # df['ph_category'] = pd.cut(df['ph'], bins=[0, 6.0, 7.5, 14], labels=['acidic', 'neutral', 'alkaline'])
    df['ph_acidic'] = (df['ph'] < 6.0).astype(int)
    df['ph_neutral'] = ((df['ph'] >= 6.0) & (df['ph'] <= 7.5)).astype(int)
    df['ph_alkaline'] = (df['ph'] > 7.5).astype(int)
    
    # Create temperature categories
    #df['temp_category'] = pd.cut(df['temperature'],
    #                            bins=[0, 15, 25, 35, 50],
    #                            labels=['cold', 'moderate', 'warm', 'hot'])
    df['temp_cold'] = (df['temperature'] < 15).astype(int)
    df['temp_moderate'] = ((df['temperature'] >= 15) & (df['temperature'] < 25)).astype(int)
    df['temp_warm'] = ((df['temperature'] >= 25) & (df['temperature'] < 35)).astype(int)
    df['temp_hot'] = (df['temperature'] >= 35).astype(int)
    
    # Create rainfall categories
    #df['rainfall_category'] = pd.cut(df['rainfall'],
     #                              bins=[0, 50, 100, 200, 300],
     #                              labels=['low', 'moderate', 'high', 'very_high'])
    df['rainfall_low'] = (df['rainfall'] < 50).astype(int)
    df['rainfall_moderate'] = ((df['rainfall'] >= 50) & (df['rainfall'] < 100)).astype(int)
    df['rainfall_high'] = ((df['rainfall'] >= 100) & (df['rainfall'] < 200)).astype(int)
    df['rainfall_very_high'] = (df['rainfall'] >= 200).astype(int)
    
    print(f"Created {df.shape[1] - 7} new features")  # 7 original features including label
    print(df.info())
    print(df.head())
    return df

def select_best_features(X, y, k=15):
    """Select the best features using statistical tests"""
    print(f"Selecting top {k} features...")
    
    # Use f_classif for feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()

    # Get feature scores
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    print("Top 10 features by score:")
    print(feature_scores.head(10))
    
    return X_selected, selected_features, selector, feature_scores


def scale_features(X_train, X_test, method='standard'):
    """Scale features using different methods"""
    print(f"Scaling features using {method} scaler...")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def main():
    """Main function for feature engineering"""
    print("Starting Feature Engineering...")
    
    # Load the original data
    try:
        # Load preprocessed data
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        with open('feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        # Convert back to DataFrame for feature engineering
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
        
        print("Loaded preprocessed data successfully!")
        
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data_analysis.py first.")
        return
    
    # Create feature combinations for training set
    X_train_enhanced = create_feature_combinations(X_train_df.copy())
    
    # Apply same transformations to test set
    X_test_enhanced = create_feature_combinations(X_test_df.copy())
    
    # Remove categorical columns for ML (keep only numeric)
    categorical_cols = ['ph_category', 'temp_category', 'rainfall_category']
    X_train_numeric = X_train_enhanced.drop(columns=categorical_cols, errors='ignore')
    X_test_numeric = X_test_enhanced.drop(columns=categorical_cols, errors='ignore')
    
     # Drop constant columns
    # X_train_numeric = drop_constant_columns(X_train_numeric)
    # X_test_numeric = X_test_numeric[X_train_numeric.columns]  # Ensure same columns


    print(f"Enhanced feature set shape: {X_train_numeric.shape}")
    
    # Feature selection
    X_train_selected, selected_features, selector, feature_scores = select_best_features(
        X_train_numeric, y_train, k=15
    )
    X_test_selected = selector.transform(X_test_numeric)
    
    # Scale the selected features
    X_train_final, X_test_final, final_scaler = scale_features(
        X_train_selected, X_test_selected, method='standard'
    )
    
    # Save enhanced data and objects
    np.save('X_train_enhanced.npy', X_train_final)
    np.save('X_test_enhanced.npy', X_test_final)
    
    with open('feature_selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    with open('final_scaler.pkl', 'wb') as f:
        pickle.dump(final_scaler, f)
    with open('selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    # Save feature scores for analysis
    feature_scores.to_csv('feature_importance.csv', index=False)
    
    print("\nFeature Engineering completed successfully!")
    print(f"Final feature set: {len(selected_features)} features")
    print(f"Selected features: {selected_features}")


if __name__ == "__main__":
    main()
