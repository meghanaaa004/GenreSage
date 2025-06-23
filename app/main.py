from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import joblib
import os
from pydub import AudioSegment
import tempfile

app = FastAPI()

# Load trained model
model = joblib.load("best_genre_classifier.pkl")

# Optional: Trivia dictionary
genre_trivia = {
    "jazz": "Jazz originated in New Orleans with roots in blues and ragtime.",
    "rock": "Rock music exploded in the 1950s with electric guitars and youth culture.",
    "pop": "Pop music is known for catchy melodies and broad appeal.",
    "classical": "Classical music includes works by Beethoven, Mozart, and Bach.",
    "hiphop": "Hip hop began in the Bronx and includes rap, DJing, and breakdancing.",
    "metal": "Metal features distorted guitars and powerful vocals.",
    "disco": "Disco dominated the 1970s with dancefloor anthems.",
    "country": "Country music blends folk, blues, and southern storytelling.",
    "blues": "Blues is a soulful genre with deep emotional roots.",
    "reggae": "Reggae originated in Jamaica and became globally famous with Bob Marley."
}

def convert_audio(file: UploadFile):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    if file.filename.endswith(".mp3"):
        audio = AudioSegment.from_file(file.file, format="mp3")
        audio.export(temp.name, format="wav")
    else:
        contents = file.file.read()
        with open(temp.name, "wb") as f:
            f.write(contents)
    return temp.name

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    features = []

    features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(tempo)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for mfcc in mfccs:
        features.append(np.mean(mfcc))
        features.append(np.var(mfcc))

    return np.array(features).reshape(1, -1)

@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    try:
        audio_path = convert_audio(file)
        features = extract_features(audio_path)
        prediction = model.predict(features)[0]
        os.remove(audio_path)

        return JSONResponse({
            "predicted_genre": prediction,
            "genre_trivia": genre_trivia.get(prediction, "No trivia available.")
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
