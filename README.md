# Medical Symptom Prediction API

A lightweight, fast chatbot API designed to predict diseases or illnesses based on user-described symptoms. This project uses **TF-IDF Vectorization** and **Cosine Similarity** to match user input against a medical dataset without the overhead of deep learning.



## Features
* **Fast Inference**: Responses are calculated using linear algebra (Cosine Similarity).
* **Confidence Threshold**: Automatically asks for clarification if symptoms are too vague or match scores are low.
* **REST API**: Built with FastAPI for easy integration with web or mobile apps.
* **Optimized for Linux**: Runs efficiently on Arch Linux (HyDE/Hyprland) or standard server environments.

## Installation

Follow these steps to set up the environment and install the required dependencies.

```bash
# Create a virtual environment
python3 -m venv init

# Activate the environment
source init/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn joblib fastapi uvicorn
