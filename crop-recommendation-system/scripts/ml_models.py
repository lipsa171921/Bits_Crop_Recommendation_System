import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationModels:
    """Class to handle multiple ML models for crop recommendation"""
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.label_encoder = None
        
    def initialize_models(self):
        """Initialize all ML models with optimized parameters"""

        print("Initializing ML models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='ovr',
                solver='liblinear'
            ),

            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42,
                probability=True
            ),

            
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='euclidean'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),

            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),

            'Naive Bayes': GaussianNB()
        }
        
        print(f"Initialized {len(self.models)} models")
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        print("\nTraining models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate accuracies
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred_test
            }
            
            print(f"  Train Accuracy: {train_accuracy:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        self.model_scores = results
        return results
    
    def find_best_model(self):
        """Find the best performing model based on test accuracy"""
        if not self.model_scores:
            print("No models trained yet!")
            return None
        
        best_name = max(self.model_scores.keys(), 
                       key=lambda x: self.model_scores[x]['test_accuracy'])
        self.best_model = (best_name, self.model_scores[best_name]['model'])
        
        print(f"\nBest Model: {best_name}")
        print(f"Test Accuracy: {self.model_scores[best_name]['test_accuracy']:.4f}")
        
        return self.best_model
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """Perform hyperparameter tuning for specified model"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'Support Vector Machine':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            model = SVC(random_state=42, probability=True)

        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)


        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_models(self, filepath_prefix='crop_model'):
        """Save all trained models"""
        print("\nSaving models...")
        
        for name, results in self.model_scores.items():
            filename = f"{filepath_prefix}_{name.replace(' ', '_').lower()}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(results['model'], f)
            print(f"Saved {name} to {filename}")
        
        # Save model comparison results
        comparison_df = pd.DataFrame({
            'Model': list(self.model_scores.keys()),
            'Train_Accuracy': [results['train_accuracy'] for results in self.model_scores.values()],
            'Test_Accuracy': [results['test_accuracy'] for results in self.model_scores.values()],
            'CV_Mean': [results['cv_mean'] for results in self.model_scores.values()],
            'CV_Std': [results['cv_std'] for results in self.model_scores.values()]
        })
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("Saved model comparison to model_comparison.csv")
    
    def load_model(self, filepath):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def predict_crop(self, model, features, feature_names=None):
        """Make crop prediction for given soil and climate features"""
        if isinstance(features, dict):
            # Convert dictionary to array
            if feature_names:
                features_array = np.array([features[name] for name in feature_names]).reshape(1, -1)
            else:
                features_array = np.array(list(features.values())).reshape(1, -1)
        else:
            features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction, probabilities

def visualize_model_comparison(model_scores):
    """Create visualizations comparing model performance"""
    print("\nCreating model comparison visualizations...")
    
    # Prepare data for plotting
    models = list(model_scores.keys())
    train_acc = [model_scores[model]['train_accuracy'] for model in models]
    test_acc = [model_scores[model]['test_accuracy'] for model in models]
    cv_scores = [model_scores[model]['cv_mean'] for model in models]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML Models Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_acc, width, label='Train Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_acc, width, label='Test Accuracy', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Train vs Test Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cross-validation scores
    axes[0, 1].bar(models, cv_scores, color='skyblue', alpha=0.8)
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('CV Score')
    axes[0, 1].set_title('Cross-Validation Scores')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Overfitting analysis (Train - Test accuracy)
    overfitting = [train_acc[i] - test_acc[i] for i in range(len(models))]
    colors = ['red' if x > 0.05 else 'green' for x in overfitting]
    axes[1, 0].bar(models, overfitting, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Train - Test Accuracy')
    axes[1, 0].set_title('Overfitting Analysis')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model ranking
    model_ranking = sorted(zip(models, test_acc), key=lambda x: x[1], reverse=True)
    ranked_models, ranked_scores = zip(*model_ranking)
    
    axes[1, 1].barh(range(len(ranked_models)), ranked_scores, color='lightcoral', alpha=0.8)
    axes[1, 1].set_yticks(range(len(ranked_models)))
    axes[1, 1].set_yticklabels(ranked_models)
    axes[1, 1].set_xlabel('Test Accuracy')
    axes[1, 1].set_title('Model Ranking by Test Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to train and evaluate all models"""
    print("Starting ML Model Training and Evaluation...")
    
    try:
        # Load preprocessed data
        X_train = np.load('X_train_enhanced.npy')
        X_test = np.load('X_test_enhanced.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        print("Loaded enhanced preprocessed data successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
    except FileNotFoundError:
        print("Enhanced data not found. Trying basic preprocessed data...")
        try:
            X_train = np.load('X_train.npy')
            X_test = np.load('X_test.npy')
            y_train = np.load('y_train.npy')
            y_test = np.load('y_test.npy')
            
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
                
            print("Loaded basic preprocessed data successfully!")
            
        except FileNotFoundError:
            print("No preprocessed data found. Please run data_analysis.py first.")
            return
    
    # Initialize and train models
    crop_models = CropRecommendationModels()
    crop_models.label_encoder = label_encoder
    crop_models.initialize_models()
    
    # Train all models
    results = crop_models.train_models(X_train, y_train, X_test, y_test)
    
    # Find best model
    best_model = crop_models.find_best_model()
    
    # Create visualizations
    visualize_model_comparison(results)
    
    # Perform hyperparameter tuning for top models
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Tune Random Forest (usually performs well)
    tuned_rf = crop_models.hyperparameter_tuning(X_train, y_train, 'Random Forest')
    if tuned_rf:
        # Evaluate tuned model
        tuned_rf.fit(X_train, y_train)
        tuned_pred = tuned_rf.predict(X_test)
        tuned_accuracy = accuracy_score(y_test, tuned_pred)
        print(f"Tuned Random Forest accuracy: {tuned_accuracy:.4f}")
        
        # Save tuned model
        with open('crop_model_tuned_random_forest.pkl', 'wb') as f:
            pickle.dump(tuned_rf, f)
    
    # Save all models
    crop_models.save_models()
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    print(f"Best Model: {best_model[0]}")
    print(f"Best Accuracy: {results[best_model[0]]['test_accuracy']:.4f}")
    
    # Show classification report for best model
    best_predictions = results[best_model[0]]['predictions']
    print(f"\nClassification Report for {best_model[0]}:")
    print(classification_report(y_test, best_predictions, 
                              target_names=label_encoder.classes_))
    
    print("\nML Model training completed successfully!")
    print("All models saved and ready for deployment.")

if __name__ == "__main__":
    main()
