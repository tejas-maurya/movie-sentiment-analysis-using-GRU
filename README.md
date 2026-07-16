# 🎬 Movie Sentiment Analysis using GRU & LSTM

A Deep Learning project that classifies movie reviews as **Positive** or **Negative** using **GRU (Gated Recurrent Unit)** and **LSTM (Long Short-Term Memory)** neural networks. This project demonstrates the complete NLP pipeline, from text preprocessing to model training and real-time sentiment prediction.

---

## 📌 Project Overview

Sentiment analysis is one of the most popular Natural Language Processing (NLP) tasks. In this project, movie reviews are processed and classified into positive or negative sentiments using Recurrent Neural Networks.

The project compares the performance of **GRU** and **LSTM** models for sequential text classification and provides an interactive interface for making predictions.

---

## 🚀 Features

- 📖 Movie Review Sentiment Classification
- 🧹 Complete NLP Text Preprocessing
- 🔤 Tokenization & Sequence Padding
- 🧠 Deep Learning using GRU and LSTM
- 📊 Model Performance Evaluation
- 📈 Training & Validation Graphs
- 💻 Interactive Streamlit Web Application
- ⚡ Real-time Sentiment Prediction

---

## 🏗️ Workflow

```
Movie Reviews
      │
      ▼
Data Preprocessing
      │
      ▼
Tokenization
      │
      ▼
Sequence Padding
      │
      ▼
Train/Test Split
      │
      ▼
Embedding Layer
      │
      ▼
GRU / LSTM Model
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Real-Time Prediction
```

---

## 🧹 Text Preprocessing

- Convert text to lowercase
- Remove HTML tags
- Remove URLs
- Remove punctuation
- Remove numbers
- Remove stopwords
- Tokenization
- Sequence Padding

---

## 🧠 Model Architecture

### Embedding Layer

Converts each word into dense vector representations.

↓

### GRU / LSTM Layer

Learns long-term dependencies from sequential text.

↓

### Dense Layer

Produces the final sentiment prediction.

↓

### Output Layer

- Positive 😊
- Negative 😞

---

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| NLP | Tokenizer, Padding, Text Preprocessing |
| Models | GRU, LSTM |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |

---

## 📂 Project Structure

```
Movie-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
├── tokenizer.pkl
├── gru_model.keras
├── lstm_model.keras
├── notebook.ipynb
├── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/movie-sentiment-analysis.git
```

Move into the project directory

```bash
cd movie-sentiment-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 💡 Example Prediction

**Input**

```
This movie was absolutely amazing! The acting was brilliant and the story kept me engaged throughout.
```

**Prediction**

```
Positive 😊
```



## 📚 What I Learned

- Natural Language Processing (NLP)
- Text preprocessing techniques
- Sequence modeling
- Working of GRU
- Working of LSTM
- Word Embeddings
- Binary Text Classification
- Deep Learning model evaluation
- Building an end-to-end AI application

---

## 🔮 Future Improvements

- Bidirectional GRU/LSTM
- Attention Mechanism
- Transformer Models (BERT)
- Hyperparameter Optimization
- Explainable AI (LIME/SHAP)
- Deploy using Docker
- Deploy on Hugging Face Spaces

---

## 👨‍💻 Author

**Tejansh Maurya**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. It motivates me to build more AI and Deep Learning projects!

---

## 📜 License

This project is licensed under the MIT License.
