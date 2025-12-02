from flask import Flask, request, jsonify, render_template, Response
from xml.parsers.expat import model
import numpy as np
from sentence_transformers import SentenceTransformer
import scann
import time
import json
import os

app = Flask(__name__)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
data = np.load("/home/keem/scaNN_Assignment/text/miniLM_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
texts = data["texts"]

# Config parameters
num_leaves = 3000
num_leaves_to_search = 1000
training_sample_size = 50000
num_segment = 2
anisotropic_quantization_threshold = 0.2
target_top_k = 500

searcher = (
    scann.scann_ops_pybind.builder(embeddings, target_top_k, "dot_product")
    .tree(num_leaves, num_leaves_to_search, training_sample_size)
    .score_ah(num_segment, anisotropic_quantization_threshold)
    .build()
)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search/scann', methods=['POST'])
def search_similar_sentences_api():
    start_time = time.time()
    
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    page = data.get('page', 1)  # Default page 1
    page_size = data.get('page_size', 100)  # Default 100 items per page
    
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    neighbors, distances = searcher.search_batched(
        q_vec, final_num_neighbors=k, pre_reorder_num_neighbors=2 * k
    )
    
    neighbors_flat = neighbors[0]
    distances_flat = distances[0]
    
    # Pagination
    total_results = len(neighbors_flat)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_results)
    
    # Only return current page
    result_texts = texts[neighbors_flat[start_idx:end_idx]].tolist()
    result_distances = distances_flat[start_idx:end_idx].tolist()
    
    total_time = time.time() - start_time
    
    results = {
        "query": query,
        "method": "scaNN",
        "texts": result_texts,
        "distances": result_distances,
        "total_results": total_results,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_results + page_size - 1) // page_size,
        "start_idx": start_idx + 1,
        "end_idx": end_idx,
        "time_ms": round(total_time * 1000, 2)
    }
    
    print(f"[ScaNN] k={k} | Page {page}/{results['total_pages']} | Time: {total_time*1000:.1f}ms")
    
    return jsonify(results)


@app.route('/search/brute-force', methods=['POST'])
def brute_force_search_api():
    start_time = time.time()
    
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    page = data.get('page', 1)
    page_size = data.get('page_size', 100)
    
    q_vec = model.encode([query], convert_to_numpy=True)[0]
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    sims = embeddings @ q_vec
    
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx_part = np.argpartition(-sims, k)[:k]
        idx = idx_part[np.argsort(-sims[idx_part])]
    
    # Pagination
    total_results = len(idx)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_results)
    
    # Only return current page
    result_texts = texts[idx[start_idx:end_idx]].tolist()
    result_sims = sims[idx[start_idx:end_idx]].tolist()
    
    total_time = time.time() - start_time
    
    results = {
        "query": query,
        "method": "brute_force",
        "texts": result_texts,
        "similarities": result_sims,
        "total_results": total_results,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_results + page_size - 1) // page_size,
        "start_idx": start_idx + 1,
        "end_idx": end_idx,
        "time_ms": round(total_time * 1000, 2)
    }
    
    print(f"[Brute] k={k} | Page {page}/{results['total_pages']} | Time: {total_time*1000:.1f}ms")
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))