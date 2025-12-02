# 🔍 ScaNN Assignment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uhgY7Q1F76mHqaqXzGNEgGmfCGvGb8Yf#scrollTo=8FkHCYtE0yHA)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> So sánh hiệu năng tìm kiếm tương đồng giữa **ScaNN** và **Brute-force** trên tập dữ liệu embedding văn bản quy mô lớn.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng chính](#tính-năng-chính)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Kết quả mong đợi](#kết-quả-mong-đợi)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Đóng góp](#đóng-góp)
- [Liên hệ](#liên-hệ)

---

## Giới thiệu

Dự án này cung cấp một bộ notebook Jupyter để so sánh hai phương pháp tìm kiếm láng giềng gần nhất (Nearest Neighbors) trên dữ liệu embedding văn bản:

| Phương pháp | Mô tả |
|-------------|-------|
| **ScaNN** | Thư viện Approximate Nearest Neighbors (ANN) của Google, tối ưu cho tốc độ cao |
| **Brute-force** | Tính cosine similarity toàn bộ dataset, dùng làm ground truth để đối chiếu |

**Mục tiêu:** Đo lường và so sánh thời gian thực thi và độ chính xác (recall) khi làm việc với dataset lớn (~500.000 vectors).

---

## Tính năng chính

- 🚀 **Hiệu năng cao**: Sử dụng ScaNN để tăng tốc truy vấn
- 📊 **So sánh chi tiết**: Đo thời gian và recall với nhiều giá trị k khác nhau
- 🔬 **Ground truth**: Brute-force làm chuẩn để đánh giá độ chính xác
- ☁️ **Chạy trên Cloud**: Hỗ trợ Google Colab, không cần cài đặt local

---

## Cấu trúc dự án

```
scaNN_Assignment/
├── 📓 scaNN.ipynb              # Notebook chính - chạy ScaNN và so sánh
├── 📄 text.csv                 # Dataset lớn (~500,000 câu văn)
├── 📦 miniLM_embeddings.npz    # File embeddings đã tính sẵn
└── 📖 README.md                # Tài liệu hướng dẫn
```

### Chi tiết các file dữ liệu

| File | Số lượng | Mô tả |
|------|----------|-------|
| `text.csv` | ~500,000 dòng | Dataset đầy đủ để benchmark |
| `miniLM_embeddings.npz` | - | Embeddings từ mô hình MiniLM |

---

## Yêu cầu hệ thống

### Chạy trên Google Colab (Khuyến nghị)
- Tài khoản Google
- Trình duyệt web hiện đại

### Chạy local
- Python 3.8+
- Các thư viện cần thiết:
  ```
  scann
  numpy
  flask
  sentence-transformers

  ```

---

## Hướng dẫn sử dụng

### Cách 1: Google Colab (Nhanh nhất)

1. **Mở notebook trên Colab:**

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uhgY7Q1F76mHqaqXzGNEgGmfCGvGb8Yf#scrollTo=8FkHCYtE0yHA)

2. **Chạy tất cả các cell:**
   - Nhấn `Runtime` → `Run all`
   - Hoặc nhấn `Ctrl+F9`

3. **Chờ kết quả:**
   - Colab sẽ tự động cài đặt các package cần thiết
   - Theo dõi output để xem kết quả benchmark

### Cách 2: Chạy local

```bash
# Clone repository
git clone https://github.com/1K-Keem/scanNN-Assignment.git
cd scaNN_Assignment

# Sử dụng WSL để chạy, tạo môi trường
python -m venv env
source env/bin/activate

# Cài đặt dependencies
pip install -m Flask/requirements.txt
```

---

## Kết quả mong đợi

Khi chạy notebook, bạn sẽ thấy:

- **Thời gian xây dựng index** cho ScaNN
- **Thời gian truy vấn** với các giá trị k khác nhau
- **So sánh recall** giữa ScaNN và Brute-force
- **Biểu đồ** trực quan hóa hiệu năng

### Ví dụ kết quả (tham khảo)

| Phương pháp | k=10 | k=50 | k=100 |
|-------------|------|------|-------|
| ScaNN | ~10ms | ~15ms | ~20ms |
| Brute-force | ~500ms | ~520ms | ~550ms |

> ⚠️ **Lưu ý:** Kết quả thực tế có thể khác tùy thuộc vào cấu hình máy và kích thước dataset.

---

## Tài liệu tham khảo

- 📖 [ScaNN GitHub Repository](https://github.com/google-research/google-research/tree/master/scann)
- 📄 [ScaNN for AlloyDB Whitepaper](https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf)
- 🔬 [Google Research Blog - ScaNN](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html)
- 📚 [Sentence Transformers Documentation](https://www.sbert.net/)

---

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Nếu bạn muốn cải thiện dự án:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

## Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo [Issue](https://github.com/1K-Keem/scanNN-Assignment/issues) trên GitHub.

---
