# 🎧 GenreSage: Audio-Based Music Genre Classifier

**GenreSage** is a machine learning-powered web API that classifies the genre of uploaded music files based on extracted audio features. Users upload `.mp3` or `.wav` files, and the system returns a genre prediction along with a trivia fact related to that genre.

Built with **FastAPI**, **Librosa**, and **Scikit-learn**, GenreSage combines audio signal processing with machine learning to create an intelligent, interactive experience.

---

## 🚀 Features

- Upload `.mp3` or `.wav` audio files  
- Predict music genre using a trained Random Forest classifier  
- Extract features like MFCCs, tempo, chroma, zero-crossing rate, and more  
- Return fun trivia for the predicted genre  
- Built with FastAPI for fast, interactive use

---

## 🧠 Tech Stack

- Python 3.10+  
- FastAPI  
- Librosa  
- Scikit-learn  
- Pydub  
- Joblib

---

## 📁 Project Structure

```

GenreSage/
│
├── app/
│   └── main.py                  # FastAPI app with prediction logic
│
├── training/
│   ├── train\_model.py           # Model training with hyperparameter tuning
│   └── train\_baseline.py        # Basic model training (no tuning)
│
├── models/
│   └── best\_genre\_classifier.pkl  # Pre-trained Random Forest model
│
├── requirements.txt             # Python package dependencies
├── README.md                    # Project overview
├── .gitignore                   # Files to exclude from Git

````

---

## 📊 Dataset

This project uses audio data derived from the **GTZAN Genre Collection**, a widely used dataset for music genre classification.

- The original dataset contains 1000 audio tracks across 10 genres (100 tracks each), all `.wav` format.  
- You can download the dataset here: [GTZAN Genre Collection on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
- Feature extraction (e.g., MFCCs, tempo) was done on raw audio files using Librosa to create CSV files used for training.  
- These CSV files are **not included** here due to size and licensing.  
- To train your own model, download the raw dataset and run the provided training scripts.

---

## ⚙️ Setup Instructions

### Install Dependencies

```bash
git clone https://github.com/meghanaaa004/GenreSage.git
cd GenreSage
pip install -r requirements.txt
````

### Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

Open your browser and go to:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API documentation.

---

## 📡 API Endpoint

### POST `/predict`

* **Request**: Upload a `.mp3` or `.wav` audio file
* **Response**:

```json
{
  "predicted_genre": "rock",
  "genre_trivia": "Rock music exploded in the 1950s with electric guitars and youth culture."
}
```

---

## 🧪 Model Training (Optional)

To retrain the model on your own dataset:

```bash
python training/train_model.py       # For hyperparameter tuning
python training/train_baseline.py    # Quick baseline training
```

The best model will be saved as `models/best_genre_classifier.pkl`.

---

## 🙌 Acknowledgments

* Dataset: [GTZAN Genre Collection on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* Libraries: Librosa, Scikit-learn, FastAPI, Pydub

---

## 📜 License

MIT License

---

## 👩‍💻 Author

**Meghana004**
[GitHub Profile](https://github.com/meghanaaa004)

