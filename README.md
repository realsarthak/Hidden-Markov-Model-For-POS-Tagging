# 🧠 HMM-based POS Tagger with Streamlit

A simple **Part-of-Speech (POS) Tagger** built using **Hidden Markov Model (HMM)** and the **Viterbi algorithm**, deployed with a user-friendly **Streamlit web interface**.

---

## 🛠️ Features

- POS tagging using HMM + Viterbi
- Trained on NLTK's Treebank corpus
- Intuitive and interactive Streamlit interface
- Supports tagging of user-input English sentences

---

## 🧑‍💻 How It Works

- **States**: Universal POS tags (NOUN, VERB, ADJ, etc.)
- **Observations**: Words from the sentence
- **Training**: Probabilities are calculated from NLTK Treebank corpus
- **Inference**: Viterbi algorithm determines the most probable sequence of tags

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hmm-pos-tagger.git
cd hmm-pos-tagger

# Install dependencies
pip install -r requirements.txt

