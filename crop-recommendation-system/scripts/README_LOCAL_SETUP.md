# Local CSV Setup Instructions

## 📁 File Placement Options

Place your `Crop_recommendation.csv` file in any of these locations:

### Option 1: Scripts Folder (Recommended)
\`\`\`
scripts/
├── Crop_recommendation.csv  ← Place here
├── data_loader.py
├── streamlit_app.py
└── ...
\`\`\`

### Option 2: Project Root
\`\`\`
project-root/
├── Crop_recommendation.csv  ← Or here
├── scripts/
│   ├── data_loader.py
│   └── ...
└── ...
\`\`\`

### Option 3: Data Subfolder
\`\`\`
scripts/
├── data/
│   └── Crop_recommendation.csv  ← Or here
├── data_loader.py
└── ...
\`\`\`

## 🚀 Running the Project

1. **Download the CSV**: Download `Crop_recommendation.csv` from Kaggle
2. **Place the file**: Put it in one of the locations above
3. **Run the project**: Execute any of the Python scripts

The data loader will automatically search all these locations and use the first CSV file it finds.

## ✅ Verification

When you run the scripts, you should see:
\`\`\`
Found CSV file at: Crop_recommendation.csv
Dataset loaded successfully from Crop_recommendation.csv! Shape: (2200, 8)
Columns: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
Unique crops: 22
\`\`\`

If the CSV is not found, the system will fall back to synthetic sample data.
