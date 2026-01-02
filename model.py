import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import joblib


print("Model files exported successfully!")
df = pd.read_csv("train_data_chatbot.csv")

df = df[df['label'] == 1.0].reset_index(drop=True)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['short_question'])

def get_response(user_input, threshold=0.3):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)

    max_similarity = np.max(similarities)
    index = np.argmax(similarities)

    if max_similarity < threshold:
        return "I'm not quite sure I understand. Could you describe your symptoms or situation in more detail?"
    else:
        return df.iloc[index]['short_answer']

print("Chatbot: Hello! Describe your symptoms or ask a medical question. (type 'quit' to exit)")
while False:
    user_query = input("You: ")
    if user_query.lower() in ['quit', 'exit', 'bye']:
        break

    response = get_response(user_query)
    print(f"Chatbot: {response}")
joblib.dump(vectorizer, 'medical_vectorizer.pkl')
joblib.dump(df, 'medical_data.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
