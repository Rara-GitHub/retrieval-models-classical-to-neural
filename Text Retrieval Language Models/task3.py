import csv
import json
import math
import time
from collections import defaultdict, Counter

from task1 import preprocess_text
from task2 import load_passages, build_inverted_index

def load_queries(file_path):
    queries = {}
    query_order = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                qid = row[0].strip()
                query = row[1].strip()
                queries[qid] = query
                query_order.append(qid)
    return queries, query_order

def compute_idf(inverted_index, N):
    idf = {}
    for term, posting in inverted_index.items():
        df = len(posting)
        idf[term] = math.log(N / df) if df > 0 else 0.0
    return idf

def build_passage_vectors(inverted_index, idf):
    passage_vectors = {}
    for term, posting in inverted_index.items():
        for pid, data in posting.items():
            tf = data["count"]
            weight = tf * idf.get(term, 0.0)
            if pid not in passage_vectors:
                passage_vectors[pid] = {}
            passage_vectors[pid][term] = weight
    for pid, vec in passage_vectors.items():
        norm = math.sqrt(sum(weight**2 for weight in vec.values()))
        if norm > 0:
            for term in vec:
                vec[term] /= norm
    return passage_vectors

def build_query_vector(query, idf, remove_stopwords=True, selective_stemming=True):
    tokens = preprocess_text(query, remove_stopwords=remove_stopwords, selective_stemming=selective_stemming)
    tf = Counter(tokens)
    vector = {}
    for term, freq in tf.items():
        if term in idf:
            vector[term] = freq * idf[term]
    norm = math.sqrt(sum(weight**2 for weight in vector.values()))
    if norm > 0:
        for term in vector:
            vector[term] /= norm
    return vector

def cosine_similarity(vec1, vec2):
    sim = 0.0
    for term, weight in vec1.items():
        if term in vec2:
            sim += weight * vec2[term]
    return sim

def retrieve_tfidf(query_vector, candidate_pids, passage_vectors):
    scores = []
    for pid in candidate_pids:
        if pid in passage_vectors:
            score = cosine_similarity(query_vector, passage_vectors[pid])
            scores.append((pid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:100]

def compute_bm25_idf(inverted_index, N):
    idf = {}
    for term, posting in inverted_index.items():
        df = len(posting)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5))
    return idf

def compute_doc_lengths(inverted_index):
    doc_lengths = defaultdict(int)
    for term, posting in inverted_index.items():
        for pid, data in posting.items():
            doc_lengths[pid] += data["count"]
    return doc_lengths

def bm25_score(query, doc_id, idf, doc_lengths, avgDL, inverted_index, k1=1.2, k2=100, b=0.75):
    query_tokens = preprocess_text(query, remove_stopwords=True, selective_stemming=True)
    qf = Counter(query_tokens)
    score = 0.0
    for term, qfreq in qf.items():
        if term not in idf:
            continue
        
        if term in inverted_index and doc_id in inverted_index[term]:
            f_td = inverted_index[term][doc_id]["count"]
        else:
            f_td = 0
        if f_td == 0:
            continue
        doc_len = doc_lengths[doc_id]

        numerator = f_td * (k1 + 1)
        denominator = f_td + k1 * (1 - b + b * (doc_len / avgDL))
        term_weight = idf[term] * (numerator / denominator)

        query_weight = (qfreq * (k2 + 1)) / (qfreq + k2)
        score += term_weight * query_weight
    return score

def retrieve_bm25(query, candidate_pids, inverted_index, idf, doc_lengths, avgDL, k1=1.2, k2=100, b=0.75):
    scores = []
    for pid in candidate_pids:
        score = bm25_score(query, pid, idf, doc_lengths, avgDL, inverted_index, k1, k2, b)
        scores.append((pid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:100]


def main():
    start_time = time.perf_counter()

    candidate_passages = load_passages("candidate-passages-top1000.tsv")
    queries, query_order = load_queries("test-queries.tsv")
    
    all_pids = set()
    for qid, passages in candidate_passages.items():
        for pid, _ in passages:
            all_pids.add(pid)
    N = len(all_pids)
    
    inverted_index = build_inverted_index(candidate_passages, remove_stopwords=True, selective_stemming=True)
    
    tfidf_idf = compute_idf(inverted_index, N)
    passage_vectors = build_passage_vectors(inverted_index, tfidf_idf)
    
    bm25_idf = compute_bm25_idf(inverted_index, N)
    doc_lengths = compute_doc_lengths(inverted_index)
    avgDL = sum(doc_lengths.values()) / float(N)
    
    tfidf_results = []
    bm25_results = []

    for qid in query_order:
        if qid not in queries:
            continue
        query_text = queries[qid]
        candidate_list = candidate_passages.get(qid, [])
        candidate_pids = [pid for pid, _ in candidate_list]
        
        query_vector = build_query_vector(query_text, tfidf_idf, remove_stopwords=True, selective_stemming=True)
        scored_tfidf = retrieve_tfidf(query_vector, candidate_pids, passage_vectors)
        for pid, score in scored_tfidf:
            tfidf_results.append((qid, pid, score))
    
        scored_bm25 = retrieve_bm25(query_text, candidate_pids, inverted_index, bm25_idf, doc_lengths, avgDL,
                                    k1=1.2, k2=100, b=0.75)
        for pid, score in scored_bm25:
            bm25_results.append((qid, pid, score))
            
    with open("tfidf.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=',', lineterminator="\n")
        for qid, pid, score in tfidf_results:
            writer.writerow([qid, pid, f"{score}"])
    
    with open("bm25.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=',', lineterminator="\n")
        for qid, pid, score in bm25_results:
            writer.writerow([qid, pid, f"{score}"])
            
    end_time = time.perf_counter()
    print(f"Integrated retrieval completed in {end_time - start_time:.4f} seconds.")
    print(f"Total rows in tfidf.csv: {len(tfidf_results)}")
    print(f"Total rows in bm25.csv: {len(bm25_results)}")

if __name__ == "__main__":
    main()
    
'''
Integrated retrieval completed in 89.8212 seconds.
Total rows in tfidf.csv: 19290
Total rows in bm25.csv: 19290
'''