# DeepGuard: Deep Learning Ransomware Detection System

A hybrid ransomware detection system using classical machine learning and deep learning models. Detects malicious PE files in real time using static metadata features from the EMBER 2018 dataset.

## 🚀 Usage Guide

### 1. Clone the Repository
```bash
git clone https://github.com/WoodenPillow/DeepGuard-Deep-Learning-Ransomware-Detection-System.git
```

### 2. Go to Deployment Folder
```bash
cd DeepGuard-Deep-Learning-Ransomware-Detection-System/Deployment
```

### 3. Create a Virtual Environment
```bash
python -m venv deepguard_env
```

### 4. Activate the Virtual Environment
##### 4.1 On Windows:
```bash
deepguard_env\Scripts\activate
```
##### 4.2 On Linux/macOS:
```bash
source deepguard_env/bin/activate
```

### 5. Install the Requirements
```bash
pip install -r requirements.txt
```

### 6. Launch the GUI Dashboard
```bash
streamlit run Home.py
```

### 7. Deactivate the Virtual Environment
```bash
deactivate
```
