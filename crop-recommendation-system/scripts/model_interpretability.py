import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text, plot_tree
import shap
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """Model interpretability and explainability analysis"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.label_encoder = None
        
    def load_models_and_data(self):
        """Load models and necessary data"""
        print("Loading models and data for interpretability analysis...")
        
        # Load best performing models
        model_files = {
            'Random Forest': 'crop_model_random_forest.pkl',
            'Gradient Boosting': 'crop_model_gradient_boosting.pkl',
            'Decision Tree': 'crop_model_decision_tree.pkl'
        }
        
        for name, file in model_files.items():
            try:
                with open(file, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Loaded {name}")
            except FileNotFoundError:
                print(f"Model {file} not found, skipping...")
        
        # Load feature names and label encoder
        try:
            with open('selected_features.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
        except FileNotFoundError:
            try:
                with open('feature_cols.pkl', 'rb') as f:
                    self.feature_names = pickle.load(f)
            except FileNotFoundError:
                self.feature_names = [f'Feature_{i}' for i in range(7)]  # Default names
        
        try:
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
        except FileNotFoundError:
            print("Label encoder not found!")
    
    def analyze_feature_importance(self, X_test, y_test):
        """Analyze feature importance using multiple methods"""
        print("\nAnalyzing feature importance...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= 3:  # Limit to 3 models for visualization
                break
                
            row = idx // 2
            col = idx % 2
            
            # Built-in feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                axes[row, col].bar(range(len(importances)), importances[indices])
                axes[row, col].set_title(f'{name} - Built-in Importance')
                axes[row, col].set_xticks(range(len(importances)))
                axes[row, col].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
                axes[row, col].set_ylabel('Importance')
        
        # Permutation importance for the best model
        if self.models:
            best_model_name = list(self.models.keys())[0]
            best_model = self.models[best_model_name]
            
            print(f"Calculating permutation importance for {best_model_name}...")
            perm_importance = permutation_importance(best_model, X_test, y_test, 
                                                   n_repeats=10, random_state=42)
            
            indices = np.argsort(perm_importance.importances_mean)[::-1]
            
            axes[1, 1].bar(range(len(perm_importance.importances_mean)), 
                          perm_importance.importances_mean[indices])
            axes[1, 1].set_title(f'{best_model_name} - Permutation Importance')
            axes[1, 1].set_xticks(range(len(perm_importance.importances_mean)))
            axes[1, 1].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
            axes[1, 1].set_ylabel('Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_decision_tree(self):
        """Visualize decision tree structure"""
        if 'Decision Tree' not in self.models:
            print("Decision Tree model not available for visualization")
            return
        
        print("\nVisualizing Decision Tree structure...")
        
        dt_model = self.models['Decision Tree']
        
        # Create tree visualization
        plt.figure(figsize=(20, 12))
        plot_tree(dt_model, 
                 feature_names=self.feature_names,
                 class_names=self.label_encoder.classes_ if self.label_encoder else None,
                 filled=True, 
                 rounded=True, 
                 fontsize=10,
                 max_depth=3)  # Limit depth for readability
        
        plt.title('Decision Tree Visualization (Max Depth = 3)', fontsize=16, fontweight='bold')
        plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Text representation
        tree_rules = export_text(dt_model, 
                                feature_names=self.feature_names,
                                max_depth=4)
        
        print("\nDecision Tree Rules (Top 4 levels):")
        print(tree_rules[:2000])  # Print first 2000 characters
        
        # Save full rules to file
        with open('decision_tree_rules.txt', 'w') as f:
            f.write(tree_rules)
        print("Full decision tree rules saved to 'decision_tree_rules.txt'")
    
    def analyze_crop_recommendations(self, X_test, y_test):
        """Analyze what conditions lead to specific crop recommendations"""
        print("\nAnalyzing crop recommendation patterns...")
        
        if not self.models or not self.label_encoder:
            print("Models or label encoder not available")
            return
        
        # Use the best model for analysis
        best_model = list(self.models.values())[0]
        predictions = best_model.predict(X_test)
        
        # Create DataFrame for analysis
        df_analysis = pd.DataFrame(X_test, columns=self.feature_names)
        df_analysis['Predicted_Crop'] = self.label_encoder.inverse_transform(predictions)
        df_analysis['Actual_Crop'] = self.label_encoder.inverse_transform(y_test)
        df_analysis['Correct'] = df_analysis['Predicted_Crop'] == df_analysis['Actual_Crop']
        
        # Analyze top crops
        top_crops = df_analysis['Predicted_Crop'].value_counts().head(8)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Feature Distributions by Predicted Crop', fontsize=16, fontweight='bold')
        
        for idx, crop in enumerate(top_crops.index):
            row = idx // 4
            col = idx % 4
            
            crop_data = df_analysis[df_analysis['Predicted_Crop'] == crop]
            
            # Select most important features for visualization
            important_features = self.feature_names[:4] if len(self.feature_names) >= 4 else self.feature_names
            
            # Create box plot for this crop
            crop_features = crop_data[important_features]
            axes[row, col].boxplot([crop_features[feat] for feat in important_features])
            axes[row, col].set_title(f'{crop} (n={len(crop_data)})')
            axes[row, col].set_xticklabels(important_features, rotation=45)
            axes[row, col].set_ylabel('Feature Value')
        
        plt.tight_layout()
        plt.savefig('crop_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary statistics
        crop_summary = df_analysis.groupby('Predicted_Crop')[self.feature_names].mean()
        crop_summary.to_csv('crop_recommendation_patterns.csv')
        
        print("Crop recommendation patterns saved to 'crop_recommendation_patterns.csv'")
        
        return df_analysis
    
    def create_prediction_examples(self, X_test, y_test, n_examples=10):
        """Create detailed examples of predictions with explanations"""
        print(f"\nCreating {n_examples} prediction examples...")
        
        if not self.models:
            print("No models available")
            return
        
        best_model = list(self.models.values())[0]
        best_model_name = list(self.models.keys())[0]
        
        # Select random examples
        indices = np.random.choice(len(X_test), n_examples, replace=False)
        
        examples = []
        
        for i, idx in enumerate(indices):
            sample = X_test[idx]
            actual = self.label_encoder.inverse_transform([y_test[idx]])[0]
            predicted = self.label_encoder.inverse_transform([best_model.predict([sample])[0]])[0]
            
            # Get prediction probabilities
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba([sample])[0]
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                top_3_crops = self.label_encoder.inverse_transform(top_3_indices)
                top_3_probs = probabilities[top_3_indices]
            else:
                top_3_crops = [predicted]
                top_3_probs = [1.0]
            
            example = {
                'Example': i + 1,
                'Actual_Crop': actual,
                'Predicted_Crop': predicted,
                'Correct': actual == predicted,
                'Features': dict(zip(self.feature_names, sample)),
                'Top_3_Predictions': list(zip(top_3_crops, top_3_probs))
            }
            
            examples.append(example)
        
        # Create detailed report
        with open('prediction_examples.txt', 'w') as f:
            f.write(f"CROP RECOMMENDATION PREDICTION EXAMPLES\n")
            f.write(f"Model: {best_model_name}\n")
            f.write("="*60 + "\n\n")
            
            for example in examples:
                f.write(f"Example {example['Example']}:\n")
                f.write(f"Actual Crop: {example['Actual_Crop']}\n")
                f.write(f"Predicted Crop: {example['Predicted_Crop']}\n")
                f.write(f"Correct: {example['Correct']}\n")
                f.write("Features:\n")
                
                for feature, value in example['Features'].items():
                    f.write(f"  {feature}: {value:.3f}\n")
                
                f.write("Top 3 Predictions:\n")
                for crop, prob in example['Top_3_Predictions']:
                    f.write(f"  {crop}: {prob:.3f}\n")
                
                f.write("\n" + "-"*40 + "\n\n")
        
        print("Detailed prediction examples saved to 'prediction_examples.txt'")
        
        return examples

def main():
    """Main function for model interpretability analysis"""
    print("Starting Model Interpretability Analysis...")
    
    # Load test data
    try:
        X_test = np.load('X_test_enhanced.npy')
        y_test = np.load('y_test.npy')
        print("Enhanced data loaded successfully!")
    except FileNotFoundError:
        print("Enhanced data not found. Using basic data...")
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    
    # Initialize interpreter
    interpreter = ModelInterpreter()
    interpreter.load_models_and_data()
    
    if len(interpreter.models) == 0:
        print("No models found! Please train models first.")
        return
    
    # Perform interpretability analysis
    interpreter.analyze_feature_importance(X_test, y_test)
    interpreter.visualize_decision_tree()
    
    # Analyze crop recommendation patterns
    analysis_df = interpreter.analyze_crop_recommendations(X_test, y_test)
    
    # Create prediction examples
    examples = interpreter.create_prediction_examples(X_test, y_test, n_examples=15)
    
    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- feature_importance_analysis.png")
    print("- decision_tree_visualization.png")
    print("- decision_tree_rules.txt")
    print("- crop_feature_distributions.png")
    print("- crop_recommendation_patterns.csv")
    print("- prediction_examples.txt")
    
    print("\nModel interpretability analysis completed successfully!")

if __name__ == "__main__":
    main()
