import re
import os
import numpy as np
import matplotlib.pyplot as plt
import nltk
from collections import Counter

try:
    stop_words = nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(file, remove_stopwords=False, selective_stemming=False, stem_condition=None):
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = file
        
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]
    
    if selective_stemming:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        if stem_condition is None:
            stem_condition = lambda token: len(token) > 3
        tokens = [stemmer.stem(token) if stem_condition(token) else token for token in tokens]
    
    return tokens


def compute_zipf_frequencies(N, s=1.0):
    ranks = np.arange(1, N + 1)
    zipf_freqs = ranks ** (-s)
    return zipf_freqs / np.sum(zipf_freqs)

def compare_empirical_to_zipf(empirical_freqs, s=1.0):
    N = len(empirical_freqs)
    zipf_freqs = compute_zipf_frequencies(N, s)
    eps = 1e-12
    mse_log = np.mean((np.log(empirical_freqs + eps) - np.log(zipf_freqs + eps)) ** 2)
    return mse_log, zipf_freqs

def compute_statistics(terms, terms_no_sw):
    vocab_size = len(terms)
    vocab_size_no_sw = len(terms_no_sw)

    sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)
    sorted_terms_no_sw = sorted(terms_no_sw.items(), key=lambda x: x[1], reverse=True)

    ranks = np.arange(1, vocab_size + 1)
    freq = np.array([freq for _, freq in sorted_terms])
    norm_freq = freq / np.sum(freq)

    ranks_nosw = np.arange(1, vocab_size_no_sw + 1)
    freq_nosw = np.array([freq for _, freq in sorted_terms_no_sw])
    norm_freq_nosw = freq_nosw / np.sum(freq_nosw)

    zipf_dist = compute_zipf_frequencies(vocab_size)
    zipf_dist_nosw = compute_zipf_frequencies(vocab_size_no_sw)

    total_diff_with = np.sum(np.abs(norm_freq - zipf_dist))
    total_diff_without = np.sum(np.abs(norm_freq_nosw - zipf_dist_nosw))

    mse_with, _ = compare_empirical_to_zipf(norm_freq)
    mse_without, _ = compare_empirical_to_zipf(norm_freq_nosw)

    return {
        "vocab_size_with": vocab_size,
        "vocab_size_without": vocab_size_no_sw,
        "total_diff_with": total_diff_with,
        "total_diff_without": total_diff_without,
        "mse_with": mse_with,
        "mse_without": mse_without,
        "ranks": ranks,
        "norm_freq": norm_freq,
        "zipf_dist": zipf_dist,
        "ranks_nosw": ranks_nosw,
        "norm_freq_nosw": norm_freq_nosw,
        "zipf_dist_nosw": zipf_dist_nosw,
    }

def plot_all_graphs(stats):
    plt.figure(figsize=(10, 6))
    plt.plot(stats["ranks"], stats["norm_freq"], marker='o', markersize=3, label='Empirical Data')
    plt.xlabel('Frequency Ranking')
    plt.ylabel('Normalised Frequency')
    plt.grid(True)
    plt.savefig(f"normal_frequency.pdf", format="pdf")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.loglog(stats["ranks"], stats["norm_freq"], marker='o', markersize=1, label='Empirical Data')
    plt.loglog(stats["ranks"], stats["zipf_dist"], marker='x', markersize=1, label="Zipf's Law (s=1)", linestyle='dashed')
    plt.xlabel('Frequency Ranking (log scale)')
    plt.ylabel('Normalised Frequency (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"log_log_frequency.pdf", format="pdf")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.loglog(stats["ranks_nosw"], stats["norm_freq_nosw"], marker='o', markersize=1, label='Empirical Data (No Stopwords)')
    plt.loglog(stats["ranks_nosw"], stats["zipf_dist_nosw"], marker='x', markersize=1, label="Zipf's Law (s=1)", linestyle='dashed')
    plt.xlabel('Frequency Ranking (log scale)')
    plt.ylabel('Normalised Frequency (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"log_log_frequency_no_sw.pdf", format="pdf")
    plt.close()

if __name__ == "__main__":
    tokens = preprocess_text("passage-collection.txt", remove_stopwords=False, selective_stemming=False)
    term_counts = Counter(tokens)
    tokens_no_sw = preprocess_text("passage-collection.txt", remove_stopwords=True, selective_stemming=False)
    term_counts_no_sw = Counter(tokens_no_sw)
    
    stats = compute_statistics(term_counts, term_counts_no_sw)
    
    print(f"Task 1 Number of tokens: {len(tokens)}")
    print(f"Vocabulary size (with stop words): {stats['vocab_size_with']}")
    print(f"Vocabulary size (without stop words): {stats['vocab_size_without']}")
    print(f"Difference: {stats['vocab_size_with'] - stats['vocab_size_without']}")
    print(f"Total absolute difference (with stop words): {stats['total_diff_with']:.4f}")
    print(f"Total absolute difference (without stop words): {stats['total_diff_without']:.4f}")
    print(f"Log-scale MSE (with stop words): {stats['mse_with']:.6f}")
    print(f"Log-scale MSE (without stop words): {stats['mse_without']:.6f}")

    plot_all_graphs(stats)
    
    
'''
Task 1 Number of tokens: 10284445
Vocabulary size (with stop words): 115698
Vocabulary size (without stop words): 115547
Difference: 151
Total absolute difference (with stop words): 0.3450
Total absolute difference (without stop words): 0.7324
Log-scale MSE (with stop words): 2.943125
Log-scale MSE (without stop words): 1.552659
'''