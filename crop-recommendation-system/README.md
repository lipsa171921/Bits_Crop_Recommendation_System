# 🌾 Crop Recommendation System

A comprehensive machine learning system for crop recommendation based on soil and climate parameters.

## 📋 Project Overview

This project implements multiple machine learning algorithms to recommend the most suitable crops for farmers based on:
- Soil nutrients (N, P, K)
- Climate conditions (temperature, humidity, rainfall)
- Soil pH level

## 🚀 Features

- **8 ML Models**: Random Forest, SVM, Logistic Regression, Decision Tree, KNN, Gradient Boosting, Neural Network, Naive Bayes
- **Ensemble Methods**: Voting and Stacking classifiers
- **Interactive Streamlit UI**: User-friendly web interface
- **Comprehensive Analysis**: Feature importance, model interpretability, statistical significance tests
- **Batch Predictions**: Upload CSV files for multiple predictions

## 🛠️ Setup Instructions for PyCharm

### 1. Project Setup
1. Open PyCharm
2. Create a new project or open existing project
3. Set the project root to the folder containing the `scripts` directory

### 2. Python Environment
1. Go to `File` → `Settings` → `Project` → `Python Interpreter`
2. Create a new virtual environment or use existing Python interpreter
3. Ensure Python 3.8+ is selected

### 3. Install Dependencies
Open PyCharm terminal and run:
\`\`\`bash
cd scripts
pip install -r requirements.txt
\`\`\`

### 4. Run Configuration Setup

#### Option A: Run Complete Pipeline
1. Right-click on `scripts/run_complete_pipeline.py`
2. Select `Run 'run_complete_pipeline'`
3. This will execute all scripts in sequence

#### Option B: Run Individual Scripts
Create run configurations for each script:

1. **Data Analysis**:
   - Script: `scripts/data_analysis.py`
   - Working directory: `scripts/`

2. **Feature Engineering**:
   - Script: `scripts/feature_engineering.py`
   - Working directory: `scripts/`

3. **ML Models Training**:
   - Script: `scripts/ml_models.py`
   - Working directory: `scripts/`

4. **Model Evaluation**:
   - Script: `scripts/model_evaluation.py`
   - Working directory: `scripts/`

#### Option C: Run Streamlit App
1. Open terminal in PyCharm
2. Navigate to scripts folder: `cd scripts`
3. Run: `streamlit run streamlit_app.py`

### 5. File Structure
\`\`\`
project/
├── scripts/
│   ├── data_analysis.py          # Data loading and EDA
│   ├── data_loader.py            # Data loading utilities
│   ├── feature_engineering.py    # Feature creation
│   ├── ml_models.py              # ML model training
│   ├── ensemble_models.py        # Ensemble methods
│   ├── model_evaluation.py       # Model evaluation
│   ├── model_interpretability.py # Model explainability
│   ├── streamlit_app.py          # Web interface
│   ├── run_complete_pipeline.py  # Complete pipeline runner
│   └── requirements.txt          # Dependencies
└── README.md
\`\`\`

## 🎯 Usage

### Quick Start
1. Run the complete pipeline:
   \`\`\`bash
   python scripts/run_complete_pipeline.py
   \`\`\`

2. Launch the web interface:
   \`\`\`bash
   cd scripts
   streamlit run streamlit_app.py
   \`\`\`

### Individual Components
- **Data Analysis**: `python data_analysis.py`
- **Model Training**: `python ml_models.py`
- **Model Evaluation**: `python model_evaluation.py`

## 📊 Model Performance

The system trains and compares 8 different ML algorithms:
- Random Forest (typically best performer)
- Support Vector Machine
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gradient Boosting
- Neural Network
- Naive Bayes

## 🌐 Web Interface Features

- **Interactive Input**: Sliders for soil and climate parameters
- **Real-time Predictions**: Instant crop recommendations
- **Model Comparison**: Performance metrics visualization
- **Batch Processing**: CSV file upload for multiple predictions
- **Data Analysis**: Dataset exploration and visualization

## 📈 Output Files

The system generates:
- Trained model files (`.pkl`)
- Performance visualizations (`.png`)
- Evaluation reports (`.csv`)
- Feature importance analysis
- Model interpretability reports

## 🔧 Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Run scripts in correct order (data_analysis.py first)
3. **Memory Issues**: Reduce dataset size or model complexity
4. **Streamlit Issues**: Check if port 8501 is available

## 📝 Notes

- The system uses synthetic data similar to the Kaggle crop recommendation dataset
- Models are optimized with hyperparameter tuning
- All visualizations are saved automatically
- The web interface provides detailed crop growing information
