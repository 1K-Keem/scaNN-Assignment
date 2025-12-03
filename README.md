# 🔍 ScaNN Search Engine

[![Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow)](https://1kzzm-scann.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HTML](https://img.shields.io/badge/HTML-58.8%25-orange)](https://github.com/1K-Keem/scaNN-Assignment)
[![Python](https://img.shields.io/badge/Python-41.2%25-blue)](https://github.com/1K-Keem/scaNN-Assignment)

> **Công cụ tìm kiếm văn bản semantic search sử dụng ScaNN (Scalable Nearest Neighbors) của Google**  
> Bài tập mở rộng môn Cấu trúc Dữ liệu và Giải thuật (DSA) - HK251 - ĐHBK TP.HCM

---

## 📑 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Demo trực tiếp](#-demo-trực-tiếp)
- [Tính năng](#-tính-năng)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Kết quả](#-kết-quả)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)
- [Tác giả](#%E2%80%8D-tác-giả)

---

## 🎯 Giới thiệu

**ScaNN Search Engine** là một hệ thống tìm kiếm ngữ nghĩa (semantic search) hiệu năng cao, được xây dựng dựa trên thư viện **ScaNN** (Scalable Nearest Neighbors) của Google Research. Dự án so sánh hiệu suất giữa phương pháp tìm kiếm xấp xỉ (Approximate Nearest Neighbors - ANN) và phương pháp brute-force truyền thống trên tập dữ liệu lớn (~500,000 vectors).

### Mục tiêu
- ⚡ Tìm kiếm văn bản nhanh chóng với độ trễ thấp
- 🎯 Độ chính xác cao với recall > 90%
- 📊 So sánh hiệu năng giữa ScaNN và Brute-force
- 🌐 Triển khai ứng dụng web thực tế

---

## 🚀 Demo trực tiếp

### 🌟 Ứng dụng chính (Hugging Face Spaces)
**[👉 Truy cập tại đây: https://1kzzm-scann.hf.space/](https://1kzzm-scann.hf.space/)**

**Tính năng:**
- Giao diện Gradio thân thiện, dễ sử dụng
- Tìm kiếm semantic với ScaNN hoặc Brute-force
- Phân trang kết quả (100 items/trang)
- Hiển thị thời gian xử lý và độ tương đồng
- Hỗ trợ tìm kiếm tiếng Việt và tiếng Anh

### 📄 GitHub Pages
**[Documentation & Report](https://1k-keem.github.io/scaNN-Assignment/)**

---

## ✨ Tính năng

- 🚀 **Tìm kiếm siêu nhanh**: ScaNN giảm thời gian tìm kiếm từ ~50ms xuống ~15ms
- 🧠 **Semantic Search**: Hiểu ngữ nghĩa câu truy vấn, không chỉ khớp từ khóa
- ⚖️ **Dual Search Mode**: Hỗ trợ cả ScaNN (nhanh) và Brute-force (chính xác 100%)
- 📊 **So sánh hiệu năng**: Đo đạc thời gian và recall chi tiết
- 🎨 **Giao diện đẹp mắt**: UI hiện đại với Gradio và Flask
- 📱 **Responsive**: Hoạt động tốt trên mọi thiết bị
- 🔄 **Phân trang thông minh**: Xử lý kết quả lớn hiệu quả

---

## 🛠️ Công nghệ sử dụng

### Backend
- **ScaNN** - Approximate Nearest Neighbors by Google
- **Sentence Transformers** - MiniLM-L6-v2 embeddings
- **NumPy** - Xử lý ma trận và vector
- **Flask** - REST API server
- **Gradio** - Interactive web interface

### Frontend
- **HTML/CSS/JavaScript** - GitHub Pages
- **Gradio UI** - Interactive components

### Deployment
- **Hugging Face Spaces** - Main demo app
- **GitHub Pages** - Documentation
- **Git LFS** - Large file storage (~721MB embeddings)

---

## 📂 Cấu trúc dự án

```
scaNN-Assignment/
│
├── 📁 Flask/                         # Flask Web Application
│   ├── app. py                        # Flask API server
│   ├── requirements. txt              # Python dependencies
│   └── templates/                    # HTML templates
│       └── index.html
│
├── 📁 HuggingFace/                   # Gradio App (deployed)
│   ├── app. py                        # Main Gradio interface
│   └── requirements.txt              # HF Space dependencies
│
├── 📁 text/                          # Dataset & Embeddings
│   └── miniLM_embeddings.npz         # Pre-computed embeddings (721MB)
│
├── 📁 Report/                        # Documentation & Reports
│   └── [Analysis reports and charts]
│
├── 📄 index.html                     # GitHub Pages landing page
├── 📄 . gitattributes                 # Git LFS configuration
├── 📄 . gitignore                     # Git ignore rules
└── 📖 README.md                      # This file
```

### Chi tiết các thành phần

| Thành phần | Mô tả | Công nghệ |
|------------|-------|-----------|
| **Flask App** | REST API cho tìm kiếm | Flask, ScaNN |
| **Gradio App** | Giao diện web tương tác | Gradio, ScaNN |
| **Embeddings** | ~500,000 vectors (384 dims) | MiniLM-L6-v2 |
| **GitHub Pages** | Trang documentation | HTML/CSS/JS |

---

## 💻 Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- 4GB RAM (khuyến nghị 8GB)
- 1GB dung lượng ổ cứng
- Linux/WSL (ScaNN không hỗ trợ Windows native)

### Cài đặt Flask App (Local)

```bash
# Clone repository
git clone https://github.com/1K-Keem/scaNN-Assignment.git
cd scaNN-Assignment/Flask

# Tạo môi trường ảo (khuyến nghị sử dụng WSL trên Windows)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy Flask server
python app.py
```

Truy cập: `http://localhost:5000`

### Cài đặt Gradio App (Local)

```bash
cd scaNN-Assignment/HuggingFace

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy Gradio app
python app.py
```

Truy cập: `http://localhost:7860`

---

## 🎮 Sử dụng

### 1. Sử dụng Demo Online (Khuyến nghị)

Truy cập **[https://1kzzm-scann.hf.space/](https://1kzzm-scann.hf.space/)**

**Các bước:**
1. Nhập câu truy vấn (tiếng Anh)
2. Chọn số lượng kết quả (k)
3. Chọn phương pháp: `scann` (nhanh) hoặc `brute-force` (chính xác)
4. Nhấn **"🔍 Tìm kiếm"**
5. Xem kết quả với điểm số tương đồng
6. Dùng **"◀ Trang trước"** / **"Trang sau ▶"** để chuyển trang

### 2. Sử dụng Flask API

```python
import requests

response = requests.post('http://localhost:5000/search', json={
    'query': 'machine learning algorithms',
    'k': 10,
    'method': 'scann'
})

results = response.json()
print(f"Found {len(results['results'])} items in {results['time']:.2f}ms")
```

### 3.  Sử dụng Python Script

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import scann

# Load model & data
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
data = np.load("text/miniLM_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
texts = data["texts"]

# Build ScaNN index
searcher = scann.scann_ops_pybind. builder(embeddings, 10, "dot_product"). tree(
    num_leaves=3000, num_leaves_to_search=1000, training_sample_size=50000
). score_ah(2, anisotropic_quantization_threshold=0.2).build()

# Search
query = "natural language processing"
q_vec = model.encode([query], normalize_embeddings=True)
neighbors, distances = searcher.search_batched(q_vec, final_num_neighbors=10)

# Results
for idx, score in zip(neighbors[0], distances[0]):
    print(f"[{score:.4f}] {texts[idx]}")
```

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Gradio UI / Flask API              │
│  (HuggingFace Spaces / Local)       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Sentence Transformer               │
│  (all-MiniLM-L6-v2)                 │
│  Input: Text → Output: 384-dim vec  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│         Search Engine               │
│  ┌─────────────┬─────────────┐      │
│  │   ScaNN     │ Brute-force │      │
│  │   (~15ms)   │  (~50ms)    │      │
│  └─────────────┴─────────────┘      │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Embeddings Database                │
│  (miniLM_embeddings.npz)            │
│  ~500,000 vectors × 384 dimensions  │
└─────────────────────────────────────┘
```

### ScaNN Configuration

```python
num_leaves = 3000                          # Số lượng leaf nodes
num_leaves_to_search = 1000                # Số leaves được tìm kiếm
training_sample_size = 50000               # Kích thước mẫu huấn luyện
num_segment = 2                            # Số segments cho quantization
anisotropic_quantization_threshold = 0.2   # Ngưỡng quantization
```

---

## 📊 Kết quả

### Benchmark Performance

| Phương pháp | k=10 | k=1000 | k=10000 | k=100000 |
|-------------|------|------|-------|-------|
| **ScaNN** | ~14-16ms | ~15-17ms | ~17-20ms | ~19-21ms |
| **Brute-force** | ~48-51ms | ~46-50ms | ~48-50ms | ~50-55ms |

### Recall Comparison

| k | Recall@k (ScaNN vs Brute-force) |
|---|----------------------------------|
| 10 | 100% |
| 100 | ~95% |
| 1000 | ~92% |
| 10000 | ~92% |
| 100000 | ~90% |

### Key Insights

- ⚡ **Tốc độ**: ScaNN nhanh hơn 3-5x so với brute-force
- 🎯 **Độ chính xác**: Duy trì recall > 90% cho mọi giá trị k
- 💾 **Bộ nhớ**: Index size ~200MB cho 500K vectors
- 🚀 **Scalability**: Xử lý tốt với dataset lớn hơn

---

## 📚 Tài liệu tham khảo

### Papers & Documentation
- 📄 [ScaNN: Efficient Vector Similarity Search](https://arxiv.org/abs/1908.10396)
- 📖 [ScaNN GitHub Repository](https://github.com/google-research/google-research/tree/master/scann)
- 🔬 [Google AI Blog - ScaNN](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html)
- 📑 [ScaNN for AlloyDB Whitepaper](https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf)

### Libraries & Tools
- 🤗 [Sentence Transformers](https://www.sbert.net/)
- 🎨 [Gradio Documentation](https://www.gradio.app/docs/)
- 🌐 [Flask Documentation](https://flask.palletsprojects.com/)

### Related Projects
- [FAISS by Facebook](https://github.com/facebookresearch/faiss)
- [Annoy by Spotify](https://github.com/spotify/annoy)
- [HNSW by Malkov & Yashunin](https://github.com/nmslib/hnswlib)

---

## 👨‍💻 Tác giả

**Trần Văn Thiên kim** ([@1K-Keem](https://github.com/1K-Keem))
**Phan Phước Thiện Quang** ([@ducklemon596](https://github.com/ducklemon596))
**Lê Đức Nguyên Khoa** ([@monoz2406](https://github.com/monoz2406))



### Extra Assignment For Honors Program
- 📚 **Môn học**: Cấu trúc Dữ liệu và Giải thuật (DSA)
- 🏫 **Trường**: Đại học Bách Khoa TP.HCM (HCMUT)
- 📅 **Học kỳ**: 251 (2025-2026)

---

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghêng!  Nếu bạn muốn cải thiện dự án:

1. Fork repository này
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3.  Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5.  Tạo Pull Request

---

## 📜 License

Dự án này được phân phối dưới giấy phép MIT.  Xem file `LICENSE` để biết thêm chi tiết.

---

<div align="center">

[🏠 Homepage](https://1k-keem.github.io/scaNN-Assignment/) | [🚀 Live Demo](https://1kzzm-scann.hf.space/) | [📖 Documentation](https://github.com/1K-Keem/scaNN-Assignment)

</div>
