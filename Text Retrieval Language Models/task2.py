import csv
import json
import time
from collections import defaultdict
from task1 import preprocess_text

def load_passages(file_path):
    passages = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 4:
                qid = row[0].strip()
                pid = row[1].strip()
                passage = row[3].strip()
                if qid not in passages:
                    passages[qid] = []
                passages[qid].append((pid, passage))
    return passages

def build_inverted_index(passages, remove_stopwords=False, selective_stemming=False):
    inverted_index = defaultdict(dict)
    for qid, passages_list in passages.items():
        for pid, passage in passages_list:
            tokens = preprocess_text(passage, remove_stopwords=remove_stopwords, selective_stemming=selective_stemming)
            for pos, token in enumerate(tokens):
                if pid not in inverted_index[token]:
                    inverted_index[token][pid] = {"count": 0, "positions": []}
                inverted_index[token][pid]["count"] += 1
                inverted_index[token][pid]["positions"].append(pos)
    return inverted_index

def save_inverted_index(inverted_index, filename="inverted_index.json"):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(inverted_index, file, indent=4)

if __name__ == "__main__":
    start_time = time.perf_counter()

    passages = load_passages("candidate-passages-top1000.tsv")
    
    inverted_index_stem = build_inverted_index(passages, remove_stopwords=True, selective_stemming=True)
    inverted_index_no_stem = build_inverted_index(passages, remove_stopwords=True, selective_stemming=False)
    
    save_inverted_index(inverted_index_stem)
    
    end_time = time.perf_counter()
    
    print(f"Inverted index built with {len(inverted_index_stem)} unique terms.")    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
'''
Inverted index built with 87254 unique terms.
Time taken: 124.2932 seconds
'''