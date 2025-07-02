# Music Genre Classification (GTZAN)

This project implements an end-to-end pipeline for automatic music genre classification using deep learning. Leveraging the popular GTZAN dataset, the system extracts audio features and trains neural networks to classify tracks into one of 10 genres.

## 🎵 Project Overview

- **Goal:** Classify music/audio files into genres using deep learning.
- **Dataset:** [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html) (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).
- **Features:** MFCCs (Mel-Frequency Cepstral Coefficients) extracted from audio files.
- **Models:** Dense Neural Network and Convolutional Neural Network (CNN) architectures implemented in TensorFlow/Keras.

## 📁 Project Structure

```
.
├── Music Genre Classification/   # Main data directory (contains genre folders)
├── extract_features.ipynb        # Feature extraction notebook
├── Evaluation (1).ipynb          # Model evaluation notebook
├── models.py                     # Model architectures
├── cnn_model.h5 / .keras         # Saved trained models
├── encoder.pkl, scaler.pkl       # Label encoder and scaler
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Music-Genre-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

- Download the [GTZAN dataset](http://marsyas.info/downloads/datasets.html) and place the genre folders under `Music Genre Classification/Train/` as:
  - `blues/`, `classical/`, `country/`, `disco/`, `hiphop/`, `jazz/`, `metal/`, `pop/`, `reggae/`, `rock/`

### 4. Feature Extraction

- Run `extract_features.ipynb` to extract MFCC features and prepare the dataset for training.

### 5. Model Training & Evaluation

- Use the provided notebooks and scripts to train models and evaluate their performance.
- Pre-trained models (`cnn_model.h5`, `.keras`) and encoders are included for quick testing.

## 🧠 Model Architectures

- **Dense Neural Network:**
  - Multiple dense layers with dropout and L2 regularization.
- **Convolutional Neural Network (CNN):**
  - 3 convolutional layers, batch normalization, dropout, and dense output layer.
  - Designed for spectrogram/MFCC input.

See `models.py` for implementation details.

## 📊 Evaluation

- Metrics: Accuracy, confusion matrix, precision, recall, F1-score.
- Evaluation notebook demonstrates model performance and visualizes results.

## 📦 Dependencies

- TensorFlow, Keras, librosa, scikit-learn, matplotlib, numpy, soundfile, and more (see `requirements.txt`).

## ✨ Results

- Achieves high accuracy on the GTZAN dataset (see evaluation notebook for details).
- Robust to different genres and audio characteristics.

## 🤝 Contributing

Pull requests and suggestions are welcome!

## 📄 License

This project is for educational and research purposes.

---

**Author:** Voora Rakesh
