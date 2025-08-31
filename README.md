# AI for Sustainable Agriculture: Leaf Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Week%201%20Active-orange.svg)](docs/project_plan.md)

## 🌱 Problem Statement

Plant diseases cause significant crop losses worldwide, leading to increased pesticide use and environmental degradation. Traditional disease detection methods are often:
- **Time-consuming**: Requires expert agronomists to inspect fields
- **Reactive**: Diseases are detected only after visible symptoms appear
- **Environmentally harmful**: Leads to excessive pesticide application

This project addresses these challenges by developing an **AI-powered early detection system** that can identify leaf diseases before they spread, enabling:
- **Targeted treatment** instead of blanket pesticide application
- **Reduced chemical usage** through precise disease identification
- **Improved crop yields** through early intervention
- **Sustainable farming practices** that protect both crops and environment

## 📊 Dataset

**PlantVillage Dataset** - A comprehensive collection of leaf images for disease classification
- **Source**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Size**: ~54,000 images across 38 classes
- **Classes**: Healthy and diseased leaves from 14 plant species
- **Format**: High-resolution RGB images with disease annotations

> **Note**: Raw data is not included in this repository due to size constraints. Please download from the Kaggle link above and place in `data/raw/PlantVillage/` directory.

## 🚀 Approach & Roadmap

### Week 1: Foundation & Baseline ✅
- [x] Repository setup and project structure
- [x] PlantVillage dataset integration
- [x] Exploratory Data Analysis (EDA)
- [x] **Baseline model**: Color histogram + Logistic Regression
- [x] Initial performance evaluation

### Week 2: Advanced Models & Explainability 🚧
- [ ] Transfer learning with MobileNetV2/EfficientNet
- [ ] Model evaluation and comparison
- [ ] Grad-CAM implementation for explainability
- [ ] Performance analysis and optimization

### Week 3: Deployment & Impact Assessment 📋
- [ ] Model optimization (TensorFlow Lite/ONNX)
- [ ] Streamlit web application demo
- [ ] Sustainability impact assessment report
- [ ] Final documentation and presentation

## 📁 Repository Structure

```
AI for Sustainable Agriculture: Leaf Disease Detection/
├── data/
│   ├── raw/           # Raw PlantVillage images (not in repo)
│   │   └── .gitkeep
│   └── processed/     # Preprocessed and augmented data
│       └── .gitkeep
├── models/            # Trained ML models and checkpoints
│   └── .gitkeep
├── reports/
│   └── figures/       # Generated charts and visualizations
│       └── .gitkeep
├── notebooks/         # Jupyter notebooks for analysis
│   └── 01_eda_dataset_overview.ipynb
├── src/              # Source code modules
│   └── baseline_colorhist.py
├── docs/             # Project documentation
│   ├── project_plan.md
│   └── project_structure.md
├── README.md         # This file
├── requirements.txt  # Python dependencies
├── .gitignore       # Git exclusions
└── LICENSE          # MIT License
```

## 🚀 Quick Start (Week 1 Baseline)

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset
1. Visit [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Download and extract to `data/raw/PlantVillage/`
3. Ensure structure: `data/raw/PlantVillage/<class_name>/<images>`

### Run Baseline Model
```bash
# Quick test with limited images
python src/baseline_colorhist.py --max_images 100

# Full dataset (may take longer)
python src/baseline_colorhist.py

# Custom data directory
python src/baseline_colorhist.py --data_dir /path/to/PlantVillage --max_images 500
```

### Expected Output
```
============================================================
Baseline Color Histogram Model for Leaf Disease Detection
============================================================
Data directory: ../data/raw/PlantVillage
Max images per class: 100
Test size: 0.2
Random state: 42
============================================================

Loading dataset...
Found 38 classes
Loading 100 images from class 'Apple___Apple_scab'
...

Extracting color histogram features...
Feature matrix shape: (3800, 48)
Features per image: 48

Training baseline model...
Baseline Model Results:
Accuracy: 0.8234 (82.34%)

Classification Report:
              precision    recall  f1-score   support
...
```

## 🌍 Sustainability Impact

### Environmental Benefits
- **Reduced Pesticide Use**: Early detection prevents disease spread, reducing need for chemical treatment
- **Targeted Application**: AI identifies specific diseases, enabling precise treatment instead of broad-spectrum pesticides
- **Water Conservation**: Healthy plants require less irrigation, reducing water waste
- **Biodiversity Protection**: Minimized chemical runoff protects soil microorganisms and beneficial insects

### Economic Benefits
- **Lower Input Costs**: Reduced pesticide and water usage decreases farming expenses
- **Higher Yields**: Early disease detection prevents crop losses, improving harvest quantities
- **Quality Improvement**: Better disease management leads to higher-grade produce
- **Market Access**: Sustainable farming practices open premium market opportunities

### Social Benefits
- **Food Security**: Improved crop yields contribute to stable food supply
- **Farmer Health**: Reduced chemical exposure improves agricultural worker safety
- **Knowledge Transfer**: AI tools democratize expert-level disease detection knowledge
- **Rural Development**: Technology adoption supports modern, profitable farming practices

## 📋 Week 1 Status Checklist

### ✅ Completed
- [x] Repository setup with proper structure and documentation
- [x] PlantVillage dataset integration and preprocessing pipeline
- [x] Exploratory Data Analysis (EDA) notebook with visualizations
- [x] Baseline model: Color histogram + Logistic Regression
- [x] Initial performance evaluation and metrics

### 🔄 In Progress
- [ ] Data preprocessing pipeline optimization
- [ ] Baseline model hyperparameter tuning
- [ ] Performance analysis and error analysis

### 📊 Current Results
- **Baseline Accuracy**: ~82% (varies by dataset size)
- **Feature Vector**: 48 dimensions (16 bins × 3 RGB channels)
- **Model**: Logistic Regression with StandardScaler
- **Data Split**: 80% train, 20% test (stratified)

## 🤝 Contributing

This project welcomes contributions! Please read our contributing guidelines and code of conduct.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd AI-for-Sustainable-Agriculture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📚 Documentation

- **Project Plan**: [docs/project_plan.md](docs/project_plan.md) - Detailed 3-week roadmap
- **Project Structure**: [docs/project_structure.md](docs/project_structure.md) - Directory explanations
- **EDA Notebook**: [notebooks/01_eda_dataset_overview.ipynb](notebooks/01_eda_dataset_overview.ipynb) - Dataset exploration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: Original dataset creators and contributors
- **Open Source Community**: Libraries and tools that made this project possible
- **Sustainable Agriculture**: Farmers and researchers working towards eco-friendly farming

---

**🌱 Together, we can build a more sustainable future through AI-powered agriculture.**

*Last updated: Week 1 of 3-week internship project*
#   s u s t a i n a b l e _ a g r i c u l t u r e e  
 