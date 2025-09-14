import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class EnsembleModels:
    """Advanced ensemble methods for crop recommendation"""
    
    def __init__(self):
        self.voting_classifier = None
        self.stacking_classifier = None
        self.weighted_ensemble = None
        
    def create_voting_ensemble(self):
        """Create a voting classifier ensemble"""
        print("Creating Voting Ensemble...")
        
        # Define base models
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm = SVC(probability=True, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Create voting classifier
        self.voting_classifier = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm),
                ('gb', gb)
            ],
            voting='soft'  # Use probability-based voting
        )
        
        return self.voting_classifier
    
    def create_stacking_ensemble(self):
        """Create a stacking classifier ensemble"""
        print("Creating Stacking Ensemble...")
        
        # Define base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42)
        
        # Create stacking classifier
        self.stacking_classifier = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
        
        return self.stacking_classifier
    
    def create_weighted_ensemble(self, models, weights):
        """Create a custom weighted ensemble"""
        print("Creating Weighted Ensemble...")
        
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = np.array(weights)
                self.weights = self.weights / self.weights.sum()  # Normalize weights
            
            def fit(self, X, y):
                for model in self.models:
                    model.fit(X, y)
                return self
            
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models])
                # Weighted majority voting
                weighted_preds = []
                for i in range(X.shape[0]):
                    sample_preds = predictions[:, i]
                    unique_preds, counts = np.unique(sample_preds, return_counts=True)
                    # Weight the counts
                    weighted_counts = {}
                    for j, pred in enumerate(sample_preds):
                        if pred not in weighted_counts:
                            weighted_counts[pred] = 0
                        weighted_counts[pred] += self.weights[j]
                    
                    # Get prediction with highest weighted count
                    best_pred = max(weighted_counts.keys(), key=lambda x: weighted_counts[x])
                    weighted_preds.append(best_pred)
                
                return np.array(weighted_preds)
            
            def predict_proba(self, X):
                if hasattr(self.models[0], 'predict_proba'):
                    probas = np.array([model.predict_proba(X) for model in self.models])
                    # Weighted average of probabilities
                    weighted_probas = np.average(probas, axis=0, weights=self.weights)
                    return weighted_probas
                else:
                    raise AttributeError("Models don't support predict_proba")
        
        self.weighted_ensemble = WeightedEnsemble(models, weights)
        return self.weighted_ensemble
    
    def train_ensembles(self, X_train, y_train, X_test, y_test):
        """Train all ensemble models"""
        print("\nTraining Ensemble Models...")
        
        results = {}
        
        # Train Voting Ensemble
        if self.voting_classifier:
            print("\nTraining Voting Ensemble...")
            self.voting_classifier.fit(X_train, y_train)
            
            vote_pred = self.voting_classifier.predict(X_test)
            vote_accuracy = accuracy_score(y_test, vote_pred)
            vote_cv = cross_val_score(self.voting_classifier, X_train, y_train, cv=5).mean()
            
            results['Voting Ensemble'] = {
                'model': self.voting_classifier,
                'accuracy': vote_accuracy,
                'cv_score': vote_cv,
                'predictions': vote_pred
            }
            
            print(f"Voting Ensemble - Test Accuracy: {vote_accuracy:.4f}, CV Score: {vote_cv:.4f}")
        
        # Train Stacking Ensemble
        if self.stacking_classifier:
            print("\nTraining Stacking Ensemble...")
            self.stacking_classifier.fit(X_train, y_train)
            
            stack_pred = self.stacking_classifier.predict(X_test)
            stack_accuracy = accuracy_score(y_test, stack_pred)
            stack_cv = cross_val_score(self.stacking_classifier, X_train, y_train, cv=3).mean()
            
            results['Stacking Ensemble'] = {
                'model': self.stacking_classifier,
                'accuracy': stack_accuracy,
                'cv_score': stack_cv,
                'predictions': stack_pred
            }
            
            print(f"Stacking Ensemble - Test Accuracy: {stack_accuracy:.4f}, CV Score: {stack_cv:.4f}")
        
        return results
    
    def save_ensembles(self):
        """Save ensemble models"""
        print("\nSaving ensemble models...")
        
        if self.voting_classifier:
            with open('voting_ensemble.pkl', 'wb') as f:
                pickle.dump(self.voting_classifier, f)
            print("Saved Voting Ensemble")
        
        if self.stacking_classifier:
            with open('stacking_ensemble.pkl', 'wb') as f:
                pickle.dump(self.stacking_classifier, f)
            print("Saved Stacking Ensemble")
        
        if self.weighted_ensemble:
            with open('weighted_ensemble.pkl', 'wb') as f:
                pickle.dump(self.weighted_ensemble, f)
            print("Saved Weighted Ensemble")

def main():
    """Main function for ensemble model training"""
    print("Starting Ensemble Model Training...")
    
    try:
        # Load data
        X_train = np.load('X_train_enhanced.npy')
        X_test = np.load('X_test_enhanced.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        print("Data loaded successfully!")
        
    except FileNotFoundError:
        print("Enhanced data not found. Using basic data...")
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    
    # Initialize ensemble models
    ensemble = EnsembleModels()
    
    # Create ensemble models
    ensemble.create_voting_ensemble()
    ensemble.create_stacking_ensemble()
    
    # Train ensembles
    results = ensemble.train_ensembles(X_train, y_train, X_test, y_test)
    
    # Save models
    ensemble.save_ensembles()
    
    # Print results
    print("\n" + "="*50)
    print("ENSEMBLE RESULTS")
    print("="*50)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy: {result['accuracy']:.4f}")
        print(f"  CV Score: {result['cv_score']:.4f}")
    
    print("\nEnsemble model training completed!")

if __name__ == "__main__":
    main()
