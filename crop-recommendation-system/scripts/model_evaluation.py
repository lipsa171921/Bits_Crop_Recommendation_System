import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve, validation_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        self.evaluation_results = {}
        self.label_encoder = None
        
    def load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        model_files = [
            'crop_model_random_forest.pkl',
            'crop_model_support_vector_machine.pkl',
            'crop_model_logistic_regression.pkl',
            'crop_model_decision_tree.pkl',
            'crop_model_k-nearest_neighbors.pkl',
            'crop_model_gradient_boosting.pkl',
            'crop_model_neural_network.pkl',
            'crop_model_naive_bayes.pkl'
        ]
        
        model_names = [
            'Random Forest',
            'Support Vector Machine',
            'Logistic Regression',
            'Decision Tree',
            'K-Nearest Neighbors',
            'Gradient Boosting',
            'Neural Network',
            'Naive Bayes'
        ]
        
        for file, name in zip(model_files, model_names):
            try:
                with open(file, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Loaded {name}")
            except FileNotFoundError:
                print(f"Model file {file} not found, skipping...")
        
        # Try to load ensemble models
        ensemble_files = ['voting_ensemble.pkl', 'stacking_ensemble.pkl']
        ensemble_names = ['Voting Ensemble', 'Stacking Ensemble']
        
        for file, name in zip(ensemble_files, ensemble_names):
            try:
                with open(file, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Loaded {name}")
            except FileNotFoundError:
                print(f"Ensemble file {file} not found, skipping...")
        
        print(f"Total models loaded: {len(self.models)}")
        
    def load_label_encoder(self):
        """Load the label encoder"""
        try:
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Label encoder loaded successfully")
        except FileNotFoundError:
            print("Label encoder not found!")
            
    def generate_predictions(self, X_test, y_test):
        """Generate predictions for all models"""
        print("\nGenerating predictions for all models...")
        
        for name, model in self.models.items():
            print(f"Generating predictions for {name}...")
            
            # Get predictions
            y_pred = model.predict(X_test)
            self.predictions[name] = y_pred
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                self.probabilities[name] = y_proba
            else:
                self.probabilities[name] = None
                
    def calculate_metrics(self, y_test):
        """Calculate comprehensive metrics for all models"""
        print("\nCalculating evaluation metrics...")
        
        for name in self.models.keys():
            y_pred = self.predictions[name]
            
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Per-class metrics
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
            
            self.evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'predictions': y_pred
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    def create_confusion_matrices(self, y_test):
        """Create confusion matrices for all models"""
        print("\nCreating confusion matrices...")
        
        n_models = len(self.models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, model) in enumerate(self.models.items()):
            row = idx // cols
            col = idx % cols
            
            y_pred = self.predictions[name]
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       ax=axes[row, col], cbar=True)
            axes[row, col].set_title(f'{name}\nAccuracy: {self.evaluation_results[name]["accuracy"]:.3f}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_roc_curves(self, y_test):
        """Create ROC curves for models with probability predictions"""
        print("\nCreating ROC curves...")
        
        # Binarize the output for multiclass ROC
        n_classes = len(np.unique(y_test))
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        plt.figure(figsize=(12, 8))
        
        for name, model in self.models.items():
            if self.probabilities[name] is not None:
                y_proba = self.probabilities[name]
                
                # Calculate ROC for each class and average
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Calculate micro-average ROC
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Plot micro-average ROC curve
                plt.plot(fpr["micro"], tpr["micro"],
                        label=f'{name} (AUC = {roc_auc["micro"]:.3f})',
                        linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multiclass Average')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_learning_curves(self, X_train, y_train, model_name='Random Forest'):
        """Create learning curves for specified model"""
        print(f"\nCreating learning curves for {model_name}...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model = self.models[model_name]
        
        # Generate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'learning_curve_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self, feature_names=None):
        """Analyze feature importance for tree-based models"""
        print("\nAnalyzing feature importance...")
        
        tree_models = ['Random Forest', 'Decision Tree', 'Gradient Boosting']
        
        plt.figure(figsize=(15, 10))
        
        for idx, model_name in enumerate(tree_models):
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    if feature_names is None:
                        feature_names = [f'Feature_{i}' for i in range(len(importances))]
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    plt.subplot(2, 2, idx + 1)
                    plt.title(f'Feature Importance - {model_name}')
                    plt.bar(range(len(importances)), importances[indices])
                    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                    plt.ylabel('Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_significance_test(self, y_test):
        """Perform statistical significance tests between models"""
        print("\nPerforming statistical significance tests...")
        
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        # Create accuracy matrix
        accuracy_matrix = np.zeros((n_models, n_models))
        p_value_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    pred1 = self.predictions[model1]
                    pred2 = self.predictions[model2]
                    
                    # Calculate per-sample accuracy
                    acc1 = (pred1 == y_test).astype(int)
                    acc2 = (pred2 == y_test).astype(int)
                    
                    # Perform paired t-test
                    t_stat, p_value = stats.ttest_rel(acc1, acc2)
                    
                    accuracy_matrix[i, j] = np.mean(acc1) - np.mean(acc2)
                    p_value_matrix[i, j] = p_value
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(accuracy_matrix, annot=True, fmt='.4f', cmap='RdBu_r',
                   xticklabels=model_names, yticklabels=model_names,
                   center=0, cbar_kws={'label': 'Accuracy Difference'})
        plt.title('Accuracy Differences Between Models')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        # Mask non-significant results (p > 0.05)
        mask = p_value_matrix > 0.05
        sns.heatmap(p_value_matrix, annot=True, fmt='.4f', cmap='viridis',
                   xticklabels=model_names, yticklabels=model_names,
                   mask=mask, cbar_kws={'label': 'P-value'})
        plt.title('Statistical Significance (p-values)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy_matrix, p_value_matrix
    
    def generate_comprehensive_report(self, y_test):
        """Generate a comprehensive evaluation report"""
        print("\nGenerating comprehensive evaluation report...")
        
        report_data = []
        
        for name in self.models.keys():
            results = self.evaluation_results[name]
            
            report_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Std_Precision': np.std(results['precision_per_class']),
                'Std_Recall': np.std(results['recall_per_class']),
                'Std_F1': np.std(results['f1_per_class'])
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Accuracy', ascending=False)
        
        # Save report
        report_df.to_csv('comprehensive_evaluation_report.csv', index=False)
        
        # Print top performers
        print("\nTop 5 Models by Accuracy:")
        print(report_df.head().to_string(index=False))
        
        # Create summary visualization
        plt.figure(figsize=(15, 10))
        
        # Metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(report_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, report_df[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Comprehensive Model Performance Comparison')
        plt.xticks(x + width*1.5, report_df['Model'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report_df

def main():
    """Main function for model evaluation"""
    print("Starting Comprehensive Model Evaluation...")
    
    # Load test data
    try:
        X_test = np.load('X_test_enhanced.npy')
        y_test = np.load('y_test.npy')
        X_train = np.load('X_train_enhanced.npy')
        y_train = np.load('y_train.npy')
        
        try:
            with open('selected_features.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        except FileNotFoundError:
            feature_names = None
            
        print("Enhanced data loaded successfully!")
        
    except FileNotFoundError:
        print("Enhanced data not found. Using basic data...")
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        
        try:
            with open('feature_cols.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        except FileNotFoundError:
            feature_names = None
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    evaluator.load_models()
    evaluator.load_label_encoder()
    
    if len(evaluator.models) == 0:
        print("No models found! Please train models first.")
        return
    
    # Generate predictions
    evaluator.generate_predictions(X_test, y_test)
    
    # Calculate metrics
    evaluator.calculate_metrics(y_test)
    
    # Create visualizations
    evaluator.create_confusion_matrices(y_test)
    evaluator.create_roc_curves(y_test)
    
    # Learning curves for best models
    if 'Random Forest' in evaluator.models:
        evaluator.create_learning_curves(X_train, y_train, 'Random Forest')
    
    # Feature importance analysis
    evaluator.feature_importance_analysis(feature_names)
    
    # Statistical significance tests
    evaluator.statistical_significance_test(y_test)
    
    # Generate comprehensive report
    report_df = evaluator.generate_comprehensive_report(y_test)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    best_model = report_df.iloc[0]
    print(f"Best Model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    
    print("\nModel evaluation completed successfully!")
    print("All evaluation results and visualizations have been saved.")

if __name__ == "__main__":
    main()
