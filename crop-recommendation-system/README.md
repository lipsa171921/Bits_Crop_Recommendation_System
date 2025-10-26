# 🌾 Smart Crop Recommendation System

A comprehensive AI-powered agricultural decision support system built with advanced machine learning algorithms and an interactive Streamlit web interface.

## 📋 Project Overview

This intelligent crop recommendation system helps farmers make data-driven decisions by analyzing:
- **Soil Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K) levels
- **Climate Conditions**: Temperature, rainfall patterns  
- **Soil Chemistry**: pH levels and soil composition
- **Environmental Factors**: Regional climate indices and growing conditions

## 🚀 Key Features

### 🤖 Advanced AI Models
- **8 ML Algorithms**: Random Forest, SVM, Logistic Regression, Decision Tree, Gradient Boosting, Neural Network, Naive Bayes and K-nearest neighbors
- **90%+ Accuracy**: Highly accurate crop predictions with confidence scoring
- **Ensemble Methods**: Voting and Stacking classifiers for improved reliability
- **Feature Engineering**: Advanced NPK ratios, climate indices, and categorical features

### 🎨 Modern Web Interface
- **Interactive Streamlit UI**: Beautiful, responsive web application
- **Real-time Predictions**: Instant crop recommendations with confidence scores
- **Interactive Visualizations**: Plotly charts and data analysis tools
- **Batch Processing**: CSV upload for multiple field predictions
- **Model Insights**: AI interpretability and decision explanations

### 📊 Comprehensive Analysis
- **Data Exploration**: Interactive dataset analysis and visualization
- **Model Comparison**: Performance metrics and algorithm rankings
- **Feature Importance**: Understanding which factors matter most
- **Growing Guides**: Detailed crop cultivation information

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or Download** the project files

2. **Install Dependencies**:
   \`\`\`bash
   cd scripts
   pip install -r requirements.txt
   \`\`\`

3. **Launch the Application**:
   \`\`\`bash
   streamlit run streamlit_app.py
   \`\`\`

4. **Open Your Browser**:
   Navigate to `http://localhost:8501`

### Alternative: Complete ML Pipeline
To train models from scratch and run full analysis:
\`\`\`bash
python run_complete_pipeline.py
\`\`\`

## 🎯 How to Use

### 🏠 Home Page - Get Recommendations
1. **Input Parameters**: Use interactive sliders to set soil and climate conditions
2. **Get Prediction**: Click "Get Smart Crop Recommendation" 
3. **View Results**: See recommended crops with confidence scores , Alternative recommendation shows top 3 recommended crops with confidence score
4. **Growing Guide**: Access detailed cultivation information

### 📊 Data Analysis
- Explore the dataset with interactive charts
- View crop distributions and feature correlations
- Understand data patterns and statistics

### 🤖 Model Comparison  
- Compare performance across different ML algorithms
- View accuracy metrics and cross-validation scores
- Understand model rankings and reliability

### 📈 Batch Predictions
- Upload CSV files with multiple field data
- Get predictions for entire regions or farm portfolios
- Download results with confidence scores

### 🎯 Model Insights
- Understand AI decision-making process
- View feature importance rankings
- Learn how different factors influence recommendations

## 📁 Project Structure

\`\`\`
crop-recommendation-system/
├── scripts/
│   ├── streamlit_app.py          # 🎨 Main Streamlit web application
│   ├── data_loader.py            # 📥 Dynamic data loading utilities
│   ├── data_analysis.py          # 📊 Data exploration and EDA
│   ├── feature_engineering.py    # ⚙️ Advanced feature creation
│   ├── ml_models.py              # 🤖 ML model training
│   ├── ensemble_models.py        # 🔗 Ensemble methods
│   ├── model_evaluation.py       # 📈 Model performance evaluation
│   ├── model_interpretability.py # 🧠 AI explainability
│   ├── run_complete_pipeline.py  # 🚀 Complete ML pipeline
│   ├── requirements.txt          # 📦 Python dependencies
├── app/
│   └── page.tsx                  # 🌐 Next.js landing page (redirects to Streamlit)
└── README.md                     # 📖 This file
\`\`\`

## 🌾 Supported Crops

The system can recommend from **40 crop types** including:
- **Cereals**: Rice, Wheat, Maize, Barley
- **Legumes**: Chickpea, Lentil, Soybean, Kidney Beans
- **Cash Crops**: Cotton, Sugarcane, Coffee, Tea
- **Fruits**: Banana, Apple, Mango, Grapes, Orange
- **Vegetables**: Potato, Tomato, Onion
- **Others**: Coconut, Jute, and more


## 🎨 Web Interface Screenshots

The Streamlit application features:
- **Modern Design**: Clean, professional interface with green agricultural theme
- **Interactive Elements**: Animated statistics, hover effects, and smooth transitions
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Data Visualizations**: Beautiful Plotly charts and interactive graphs

## 🔧 Technical Details

### Machine Learning Pipeline
1. **Data Loading**: Dynamic CSV loading  with automatic feature detection
2. **Feature Engineering**: NPK ratios, climate indices, categorical encoding
3. **Model Training**: 8 algorithms with hyperparameter optimization
4. **Evaluation**: Cross-validation, confusion matrices, classification reports
5. **Deployment**: Real-time prediction API through Streamlit

### Advanced Features
- **Dynamic Feature Detection**: Automatically adapts to different datasets
- **Enhanced Feature Engineering**: Creates 15+ derived features from basic inputs
- **Model Interpretability**: SHAP values and feature importance analysis
- **Confidence Scoring**: Probability-based recommendation confidence
- **Batch Processing**: Efficient handling of multiple predictions

## 🚀 Deployment Options

### Local Development
\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

### Production Deployment (Not Done)
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Container-based deployment
- **AWS/GCP**: Cloud platform deployment
- **Docker**: Containerized deployment

## 🙏 Acknowledgments

- Built with ❤️ for sustainable agriculture
- Powered by Streamlit, scikit-learn, and Plotly
- Designed to help farmers make better crop decisions
- Contributing to global food security through AI

---

**🌱 Happy Farming! 🌱**

