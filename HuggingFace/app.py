import uuid
import time
import html
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
import scann

# ---------- LOAD MODEL & DATA ----------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
data = np.load("text/miniLM_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
texts = data["texts"]

# ---------- SCANN INDEX ----------
num_leaves = 3000
num_leaves_to_search = 1000
training_sample_size = min(50000, len(embeddings))
num_segment = 2
anisotropic_quantization_threshold = 0.2
target_top_k = 500

searcher = (
    scann.scann_ops_pybind.builder(embeddings, target_top_k, "dot_product")
    .tree(num_leaves, num_leaves_to_search, training_sample_size)
    .score_ah(num_segment, anisotropic_quantization_threshold)
    .build()
)

PAGE_SIZE = 100

# Lưu kết quả tìm kiếm theo session_id: sid -> (indices, scores, msg)
RESULTS_STORE = {}


# ---------- BACKEND SEARCH (KHÔNG PHÂN TRANG) ----------
def backend_search(query, k, method):
    query = (query or "").strip()
    if not query:
        return None, None, 0.0, "⚠ Vui lòng nhập từ khóa"

    try:
        k = int(k)
        if k < 1:
            k = 10
    except Exception:
        k = 10

    start = time.time()

    if method == "scann":
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        neighbors, distances = searcher.search_batched(
            q,
            final_num_neighbors=k,
            pre_reorder_num_neighbors=min(2 * k, len(embeddings)),
        )
        indices = neighbors[0]
        scores = distances[0]
    else:
        q = model.encode([query], convert_to_numpy=True)[0]
        q = q / np.linalg.norm(q)
        sims = embeddings @ q

        if k >= len(sims):
            indices = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            indices = part[np.argsort(-sims[part])]
        scores = sims[indices]

    elapsed = (time.time() - start) * 1000.0
    msg = f"🔥 Tìm thấy {len(indices)} kết quả trong {elapsed:.1f}ms"
    return indices, scores, elapsed, msg


# ---------- RENDER 1 PAGE ----------
def render_page(indices, scores, page, msg):
    if indices is None or scores is None:
        html_content = f"""
        <div class="results-header">
            <span class="header-msg">{msg}</span>
        </div>
        """
        dd = gr.update(choices=[], value=None, interactive=False)
        return html_content, "Trang 0/0", 1, gr.update(interactive=False), gr.update(interactive=False), dd

    total = len(indices)
    if total == 0:
        html_content = f"""
        <div class="results-header">
            <span class="header-msg">{msg}</span>
            <span class="header-range">(Không có kết quả)</span>
        </div>
        <div class="results-list">
            <div class="result-item">
                <div class="result-text">Không có kết quả để hiển thị.</div>
            </div>
        </div>
        """
        dd = gr.update(choices=[], value=None, interactive=False)
        return html_content, "Trang 0/0", 1, gr.update(interactive=False), gr.update(interactive=False), dd

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total)

    html_parts = [
        f"""
        <div class="results-header">
            <span class="header-msg">{msg}</span>
            <span class="header-range">(Hiển thị {start_idx + 1} - {end_idx} / {total})</span>
        </div>
        <div class="results-list">
        """
    ]

    for i in range(start_idx, end_idx):
        idx = int(indices[i])
        score = float(scores[i])
        rank = i + 1
        safe_text = html.escape(str(texts[idx]))
        html_parts.append(f"""
        <div class="result-item">
            <div class="result-meta">
                <span class="rank-badge">#{rank}</span>
                <span class="score-badge">Score: {score:.4f}</span>
            </div>
            <div class="result-text">{safe_text}</div>
        </div>
        """)

    html_parts.append("</div>")
    html_content = "".join(html_parts)

    page_label = f"Trang {page}/{total_pages}"
    prev_state = gr.update(interactive=(page > 1))
    next_state = gr.update(interactive=(page < total_pages))
    dd_state = gr.update(
        choices=[str(i) for i in range(1, total_pages + 1)],
        value=str(page),
        interactive=True,
    )

    return html_content, page_label, page, prev_state, next_state, dd_state


# ---------- HANDLERS ----------
def on_search(query, k, method):
    indices, scores, elapsed, msg = backend_search(query, k, method)

    if indices is None:
        sid = None
        html_content, page_label, page, prev_s, next_s, dd = render_page(None, None, 1, msg)
        return sid, html_content, page_label, page, prev_s, next_s, dd

    sid = uuid.uuid4().hex
    RESULTS_STORE[sid] = (indices, scores, msg)

    html_content, page_label, page, prev_s, next_s, dd = render_page(indices, scores, 1, msg)
    return sid, html_content, page_label, page, prev_s, next_s, dd


def on_page_change(session_id, current_page, direction):
    if not session_id or session_id not in RESULTS_STORE:
        return gr.update(), "Trang 0/0", 1, gr.update(interactive=False), gr.update(interactive=False), gr.update(choices=[], value=None, interactive=False)

    indices, scores, msg = RESULTS_STORE[session_id]
    new_page = current_page + direction
    return render_page(indices, scores, new_page, msg)


def on_dropdown_change(session_id, target_page_str):
    if not session_id or session_id not in RESULTS_STORE or not target_page_str:
        return gr.update(), "Trang 0/0", 1, gr.update(interactive=False), gr.update(interactive=False), gr.update(choices=[], value=None, interactive=False)

    try:
        target_page = int(target_page_str)
    except Exception:
        target_page = 1

    indices, scores, msg = RESULTS_STORE[session_id]
    return render_page(indices, scores, target_page, msg)


# ---------- CSS ----------
css = """
.container { max-width: 1200px; margin: auto; }

#results-box {
    height: 600px !important;
    overflow-y: auto !important;
    background-color: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    padding: 0 !important;
    position: relative !important;
}

.results-header {
    position: sticky !important;
    top: 0 !important;
    background-color: #111827 !important;
    border-bottom: 2px solid #374151 !important;
    padding: 12px 15px !important;
    z-index: 50 !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}

.header-msg { color: #34d399 !important; font-weight: bold !important; }
.header-range { color: #9ca3af !important; font-size: 0.9em !important; }

.results-list { padding: 10px !important; }

.result-item {
    background-color: #1f2937 !important;
    border-bottom: 1px solid #374151 !important;
    padding: 10px !important;
    margin-bottom: 0 !important;
    content-visibility: auto !important;
    contain-intrinsic-size: 80px !important;
}

.result-item:hover { background-color: #374151 !important; }

.result-meta {
    display: flex !important;
    gap: 10px !important;
    margin-bottom: 5px !important;
    align-items: center !important;
}

.rank-badge  { color: #60a5fa !important; font-weight: bold !important; font-size: 0.9em !important; }
.score-badge { color: #34d399 !important; font-family: monospace !important; font-size: 0.9em !important; }

.result-text {
    color: #e5e7eb !important;
    line-height: 1.5 !important;
    font-size: 0.95em !important;
}

.pagination-bar {
    background-color: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    margin-top: 10px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 12px !important;
}

.pg-btn {
    padding: 6px 12px !important;
    background-color: #374151 !important;
    color: #e5e7eb !important;
    border-radius: 6px !important;
}

#page-label {
    color: #e5e7eb !important;
    white-space: nowrap !important;
}

#page-select {
    background-color: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    color: #e5e7eb !important;
    padding: 4px 8px !important;
}

/* tắt hiệu ứng dim khi component pending */
.gradio-container .pending, .gradio-container .pending * {
    opacity: 1 !important;
    filter: none !important;
}
"""

# ---------- UI ----------
with gr.Blocks(title="ScaNN Search Engine", theme=gr.themes.Soft(), css=css) as demo:
    state_sid = gr.State(None)
    state_page = gr.State(1)

    gr.Markdown("# ⚡ ScaNN Fast Semantic Search")

    with gr.Row():
        with gr.Column(scale=4):
            txt_query = gr.Textbox(
                label="Nội dung tìm kiếm",
                placeholder="Nhập văn bản cần tìm...",
                lines=1,
            )
            gr.Examples(
                examples=[
                    "A boy is playing",
                    "Surfing",
                    "VietNam",
                    "Something good for your health",
                    "The weather is not good today",
                ],
                inputs=txt_query,
                label="Gợi ý tìm kiếm (Click để thử)",
            )
        with gr.Column(scale=1):
            num_k = gr.Number(label="Số lượng kết quả (Top-K)", value=10, minimum=1, step=1)

    with gr.Row():
        btn_scann = gr.Button("🚀 Tìm kiếm nhanh (ScaNN)", variant="primary")
        btn_brute = gr.Button("🐢 Tìm kiếm chính xác (Brute Force)")

    gr.Markdown("### Kết quả tìm kiếm")

    html_results = gr.HTML(elem_id="results-box")

    with gr.Row(elem_classes=["pagination-bar"]):
        btn_prev = gr.Button("◀ Trước", size="sm", interactive=False)
        lbl_page_info = gr.HTML("Trang 0/0", elem_id="page-label")
        dd_pages = gr.Dropdown(
            choices=[],
            value=None,
            interactive=False,
            show_label=False,
            elem_id="page-select",
            scale=0,
            min_width=100,
        )
        btn_next = gr.Button("Sau ▶", size="sm", interactive=False)

    # SEARCH
    btn_scann.click(
        fn=lambda q, k: on_search(q, k, "scann"),
        inputs=[txt_query, num_k],
        outputs=[state_sid, html_results, lbl_page_info, state_page, btn_prev, btn_next, dd_pages],
    )
    btn_brute.click(
        fn=lambda q, k: on_search(q, k, "brute"),
        inputs=[txt_query, num_k],
        outputs=[state_sid, html_results, lbl_page_info, state_page, btn_prev, btn_next, dd_pages],
    )

    # PAGINATION BUTTONS
    btn_prev.click(
        fn=lambda sid, pg: on_page_change(sid, pg, -1),
        inputs=[state_sid, state_page],
        outputs=[html_results, lbl_page_info, state_page, btn_prev, btn_next, dd_pages],
        show_progress="hidden",
    )
    btn_next.click(
        fn=lambda sid, pg: on_page_change(sid, pg, 1),
        inputs=[state_sid, state_page],
        outputs=[html_results, lbl_page_info, state_page, btn_prev, btn_next, dd_pages],
        show_progress="hidden",
    )

    # DROPDOWN PAGE
    dd_pages.change(
        fn=lambda sid, tgt: on_dropdown_change(sid, tgt),
        inputs=[state_sid, dd_pages],
        outputs=[html_results, lbl_page_info, state_page, btn_prev, btn_next, dd_pages],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()