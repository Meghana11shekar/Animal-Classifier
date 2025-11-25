🐾 Animal Prediction using Deep Learning

An AI-powered multi-class animal image classifier built using TensorFlow, MobileNetV3, and Streamlit.
The model predicts the animal in an uploaded image with confidence scores and ranked outputs, packaged inside a clean and responsive web interface.

🚀 Features

📷 Real-time image upload and instant prediction

🤖 MobileNetV3-based deep learning model

📊 Displays top predictions with confidence percentages

🔄 Data augmentation + fine-tuning for higher accuracy

🎨 Streamlit UI with enhanced custom CSS styling

🦁 Trained on the Animal-10 dataset


🛠️ Tech Stack
Python 3.10

TensorFlow / Keras

MobileNetV3

NumPy

OpenCV

Streamlit


📂 Project Structure
Animal-Classifier/
│── app.py                         # Streamlit frontend
│── model/                         # Trained MobileNetV3 model
│── helpers/                       # Preprocessing utilities
│── static/                        # CSS or sample images
│── requirements.txt
│── README.md

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Meghana11shekar/Animal-Classifier.git
cd Animal-Classifier

2️⃣ Create and activate Conda environment (recommended)
conda create -n animal python=3.10 -y
conda activate animal

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the app
streamlit run app.py


The app will open at:

http://localhost:8501

🧠 Model Training (Optional)

If you want to retrain the model:

python train.py

Dataset can be extended or modified inside the /data folder.

🔮 Future Enhancements

Add more animal classes

Improve model explainability using Grad-CAM

Deploy on Streamlit Cloud, Render, or HuggingFace Spaces

Add image preprocessing visualizations

🤝 Contributions

Contributions, issues, and feature requests are welcome!

👩‍💻 Author

Meghana Shekar
🔗 GitHub: Meghana11shekar
