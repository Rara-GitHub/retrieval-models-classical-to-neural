# Project 1: Text Retrieval Language Models
This project implements and evaluates classic query likelihood language models for information retrieval, based on unigram representations. 
It includes detailed text preprocessing and analysis of Zipf’s Law, along with an inverted index and three smoothed language models.

Query Likelihood Language Models:
- Full text preprocessing pipeline (tokenisation, stemming, etc.)
- Zipf's Law analysis of word frequency distributions
- Inverted index construction
- Smoothed query likelihood models: **Laplace**, **Lidstone**, **Dirichlet**
- Ranking evaluation and comparison

📄 Includes: `report.pdf`, source code, and sample outputs


# Project 2: Neural Re-ranking for Information Retrieval
This project investigates neural passage re-ranking models for text retrieval, progressively advancing from classical BM25 to neural networks including MLP and Siamese BiLSTM.

Learning-to-Rank Models:
- Baseline: **BM25**
- Shallow model: **Logistic Regression** on GloVe embeddings
- Tree model: **LambdaMART (XGBoost)**
- Neural models: **MLP** and **Siamese BiLSTM**
- Evaluation: Mean Average Precision (mAP), Normalised DCG (nDCG)

📄 Includes: `report.pdf`, training scripts, and output rankings
