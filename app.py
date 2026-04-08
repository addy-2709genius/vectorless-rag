import os
import streamlit as st
from dotenv import load_dotenv
from ingestion import process_file
from retrieval import build_index, search
from reranker import rerank
from llm import stream_answer
from utils import format_context, truncate_context

load_dotenv()

st.set_page_config(layout="wide", page_title="Vectorless RAG", page_icon="🔍", initial_sidebar_state="expanded")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="collapsedControl"] {display: none !important;}

    section[data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
        transform: none !important;
        visibility: visible !important;
        display: block !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .stButton > button {
        background-color: #22c55e !important;
        color: #000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover {
        background-color: #16a34a !important;
        color: #fff !important;
    }

    div[data-testid="stSidebarContent"] {
        background-color: #f8fafb;
        border-right: 1px solid #e2e8f0;
    }

    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #e2e8f0;
    }

    .stSelectbox > div > div {
        border-radius: 6px;
    }

    .stExpander {
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
    }

    .stChatMessage {
        border-radius: 10px;
    }

    .stAlert {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None

with st.sidebar:
    st.markdown("<h2 style='font-size:1.1rem; font-weight:700; color:#1a1a1a; margin-bottom:0;'>Vectorless RAG</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.72rem; color:#64748b; margin-top:2px; text-transform:uppercase; letter-spacing:0.8px;'>BM25 + Reranker + Groq LLM</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<p style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;'>Documents</p>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

    with st.expander("Chunking Settings", expanded=False):
        chunk_size = st.slider("Chunk size (words)", 100, 500, 300)
        top_k = st.slider("BM25 top-k", 10, 50, 20)
        top_n = st.slider("Rerank top-n", 3, 10, 5)

    if st.button("Index Documents", use_container_width=True):
        if uploaded_files:
            all_chunks = []
            progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                st.caption(f"Processing {f.name}...")
                chunks = process_file(f, chunk_size=chunk_size)
                all_chunks.extend(chunks)
                progress.progress((i + 1) / len(uploaded_files))
            st.session_state.chunks = all_chunks
            st.session_state.bm25_index = build_index(all_chunks)
            st.success(f"{len(all_chunks)} chunks indexed from {len(uploaded_files)} file(s)")
        else:
            st.warning("Upload at least one file first.")

    st.divider()
    st.markdown("<p style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;'>Groq Settings</p>", unsafe_allow_html=True)

    api_key = st.text_input("API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
        except:
            api_key = os.getenv("GROQ_API_KEY", "")

    model = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ], label_visibility="collapsed")

    with st.expander("Advanced Settings", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages and st.session_state.bm25_index is None:
    st.components.v1.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: #ffffff; overflow: hidden; }
  canvas { position: fixed; top: 0; left: 0; width: 100% !important; height: 100% !important; }

  .page {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
    pointer-events: none;
    padding: 2rem;
  }

  .left {
    flex: 1.1;
    padding-right: 2rem;
  }

  .right {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .byline {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }

  h1 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #16a34a;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.5rem;
  }

  .subtitle {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.95rem;
    color: #64748b;
    line-height: 1.6;
    margin-bottom: 1rem;
    max-width: 420px;
  }

  .tags { margin-bottom: 1.2rem; }

  .tag {
    display: inline-block;
    margin: 0.2rem 0.2rem 0 0;
    padding: 0.22rem 0.7rem;
    border: 1px solid #22c55e;
    border-radius: 999px;
    font-size: 0.7rem;
    color: #22c55e;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  .hint {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.78rem;
    color: #94a3b8;
    animation: pulse 2.5s infinite;
  }

  .card {
    background: rgba(255,255,255,0.88);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(6px);
  }

  .card-title {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .card-text {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.65;
  }

  .arch {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    flex-wrap: wrap;
  }

  .arch-step {
    background: #f8fafb;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.3rem 0.55rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.7rem;
    color: #1a1a1a;
    font-weight: 500;
  }

  .arch-step.green {
    background: #f0fdf4;
    border-color: #22c55e;
    color: #16a34a;
    font-weight: 600;
  }

  .arch-arrow {
    font-size: 0.7rem;
    color: #cbd5e1;
  }

  .no-list { display: flex; flex-direction: column; gap: 0.3rem; }

  .no-row {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.75rem;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .no-badge {
    background: #fef2f2;
    color: #ef4444;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }
</style>
</head>
<body>
<canvas id="nn-canvas"></canvas>
<div class="page">
  <div class="left">
    <div class="byline">A product by Aaditya Raj Soni</div>
    <h1>Vectorless RAG</h1>
    <p class="subtitle">Document Q&A without any vector database or embedding model. Pure retrieval, pure reasoning.</p>
    <div class="tags">
      <span class="tag">BM25 Retrieval</span>
      <span class="tag">Cross-Encoder Reranking</span>
      <span class="tag">Groq LLM</span>
      <span class="tag">Zero Infrastructure Cost</span>
    </div>
    <div class="hint">← Upload a document in the sidebar to begin</div>
  </div>

  <div class="right">
    <div class="card">
      <div class="card-title">What is Vectorless RAG?</div>
      <div class="card-text">
        Traditional RAG converts text into embeddings and stores them in a vector database — requiring expensive infrastructure. Vectorless RAG replaces this with <strong>BM25</strong> sparse retrieval and a <strong>cross-encoder reranker</strong>, achieving comparable accuracy at zero infra cost.
      </div>
    </div>

    <div class="card">
      <div class="card-title">System Architecture</div>
      <div class="arch">
        <div class="arch-step green">PDF / TXT</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">Chunker</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step green">BM25 Index</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">Top-20 Chunks</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step green">Reranker</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">Top-5 Chunks</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step green">Groq LLM</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">Answer</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">No Infrastructure Required</div>
      <div class="no-list">
        <div class="no-row"><span class="no-badge">NO</span> Vector database (Pinecone, Weaviate, Chroma)</div>
        <div class="no-row"><span class="no-badge">NO</span> Embedding model or API</div>
        <div class="no-row"><span class="no-badge">NO</span> LangChain or LlamaIndex</div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const canvas = document.getElementById('nn-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setClearColor(0xffffff, 1);

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0xffffff, 0.028);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 25);
resize();

scene.add(new THREE.AmbientLight(0x888888));
const light = new THREE.PointLight(0x22c55e, 3);
light.position.set(5, 5, 10);
scene.add(light);

const layers = [
  { n: 4, x: -10 },
  { n: 8, x: -5 },
  { n: 12, x: 0 },
  { n: 8, x: 5 },
  { n: 4, x: 10 }
];
const nodes = [];
const geometry = new THREE.SphereGeometry(0.3, 16, 16);

layers.forEach(layer => {
  const layerNodes = [];
  for (let i = 0; i < layer.n; i++) {
    const material = new THREE.MeshStandardMaterial({
      color: 0x22c55e,
      emissive: 0x22c55e,
      emissiveIntensity: 0.3
    });
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.set(
      layer.x,
      (i - layer.n / 2) * 2,
      (Math.random() - 0.5) * 4
    );
    scene.add(sphere);
    layerNodes.push(sphere);
  }
  nodes.push(layerNodes);
});

for (let i = 0; i < nodes.length - 1; i++) {
  nodes[i].forEach(a => {
    nodes[i + 1].forEach(b => {
      const geo = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
      const mat = new THREE.LineBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.1 });
      scene.add(new THREE.Line(geo, mat));
    });
  });
}

function animate() {
  requestAnimationFrame(animate);
  scene.rotation.y += 0.0015;
  renderer.render(scene, camera);
}
animate();
window.addEventListener('resize', resize);
</script>
</body>
</html>
""", height=650, scrolling=False)

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if st.session_state.bm25_index is None:
    st.stop()

if not api_key:
    st.error("Groq API key missing. Get one free at https://console.groq.com")
    st.stop()

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Retrieving..."):
        bm25_results = search(query, st.session_state.bm25_index, st.session_state.chunks, top_k=top_k)

    if not bm25_results:
        st.warning("No relevant chunks found.")
        st.stop()

    with st.spinner("Reranking..."):
        reranked = rerank(query, bm25_results, top_n=top_n)

    with st.expander("View retrieved chunks"):
        tab1, tab2 = st.tabs(["BM25 top results", "Reranked top results"])
        with tab1:
            for c in bm25_results:
                st.markdown(f"**Score:** `{c['bm25_score']:.4f}` | **Source:** `{c['source_file']}` | **Chunk:** `#{c['chunk_index']}`")
                st.caption(c["text"][:120] + "...")
                st.divider()
        with tab2:
            for c in reranked:
                st.markdown(f"**Rerank Score:** `{c['rerank_score']:.4f}` | **Source:** `{c['source_file']}` | **Chunk:** `#{c['chunk_index']}`")
                st.write(c["text"])
                st.divider()

    safe_chunks = truncate_context(reranked)
    context = format_context(safe_chunks)

    response, latency, used_model = stream_answer(query, context, api_key, model, temperature)

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown(f"<p style='color:#22c55e; font-size:0.78rem;'><code>{used_model}</code> &nbsp;|&nbsp; {latency}s</p>", unsafe_allow_html=True)