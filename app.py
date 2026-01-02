from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = joblib.load('medical_vectorizer.pkl')
df = joblib.load('medical_data.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

app = FastAPI()

class SymptomRequest(BaseModel):
    symptom: str

@app.post("/predict")
async def predict(request: SymptomRequest):
    user_input = request.symptom
    threshold = 0.3

    query_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(query_vec, tfidf_matrix)

    max_sim = np.max(similarities)
    best_idx = np.argmax(similarities)

    if max_sim < threshold:
        return {
            "status": "inconclusive",
            "answer": "I need more details about your symptoms to give a clear answer.",
            "confidence": float(max_sim)
        }

    return {
        "status": "success",
        "answer": df.iloc[best_idx]['short_answer'],
        "tags": df.iloc[best_idx]['tags'],
        "confidence": float(max_sim)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
