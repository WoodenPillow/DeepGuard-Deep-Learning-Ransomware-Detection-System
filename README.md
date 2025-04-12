git clone https://github.com/WoodenPillow/DeepGuard-Hybrid-Ransomware-Detection-System.git
cd Deployment
python -m venv deepguard_env
deepguard_env\Scripts\activate
pip install -r requirements.txt
streamlit run GUI.py
