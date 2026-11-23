"""Interfaz de demostración (Streamlit) que consume la API FastAPI.

Ejecutar:  streamlit run frontend/streamlit_app.py
Requiere la API levantada (uvicorn app.api.main:app).
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("HEALTHRAG_API_URL", "http://localhost:8000")
API_KEY = os.getenv("HEALTHRAG_UI_API_KEY", "")

st.set_page_config(page_title="HealthRAG", page_icon="🩺", layout="wide")
st.title("HealthRAG — consulta de guías clínicas")
st.caption("Hipertensión arterial y salud cardiovascular · respuestas con fuentes verificables")

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Arquitectura", ["multiagent", "advanced", "basic"],
                    format_func={"multiagent": "RAG avanzado + verificación multiagente",
                                 "advanced": "RAG avanzado (híbrido + reranking)",
                                 "basic": "RAG básico (vectorial)"}.get)
    top_k = st.slider("Fragmentos de evidencia (top-k)", 1, 10, 5)
    use_cache = st.checkbox("Usar caché", value=True)
    show_trace = st.checkbox("Mostrar diálogo entre agentes", value=True)
    try:
        info = httpx.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API OK · {info['documents']} documentos · {info['chunks']} fragmentos")
        st.caption(f"Embeddings: {info['embedding_model']}  \nReranker: {info['reranker']}  \n"
                   f"LLM: {info['llm_provider']}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo conectar con la API en {API_URL}: {exc}")

question = st.text_area("Consulta", placeholder="¿Cuánta sal se recomienda consumir por día?",
                        max_chars=600)

if st.button("Consultar", type="primary", disabled=not question.strip()):
    with st.spinner("Recuperando evidencia y verificando..."):
        try:
            resp = httpx.post(f"{API_URL}/query", timeout=180,
                              headers={"X-API-Key": API_KEY} if API_KEY else {},
                              json={"question": question, "mode": mode, "top_k": top_k,
                                    "use_cache": use_cache, "include_trace": show_trace})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"Error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error de conexión: {exc}")
            st.stop()

    if data.get("blocked"):
        st.error(data["answer"])
    elif data["abstained"]:
        st.warning(data["answer"])
    else:
        st.markdown(data["answer"])
    for w in data.get("warnings", []):
        st.info(w)

    cols = st.columns(5)
    cols[0].metric("Latencia", f"{data['latency_ms'] / 1000:.2f} s")
    cols[1].metric("Tokens", data.get("tokens", 0))
    cols[2].metric("Costo est.", f"US$ {data.get('cost_usd', 0):.4f}")
    cols[3].metric("Respaldo", f"{(data.get('supported_ratio') or 0) * 100:.0f} %"
                   if data.get("supported_ratio") is not None else "—")
    cols[4].metric("Caché", "sí" if data.get("cache_hit") else "no")

    if data.get("sources"):
        st.subheader("Fuentes citadas")
        for s in data["sources"]:
            with st.expander(f"[{s['eid']}] {s.get('citation') or s['doc_id']}"):
                st.write(s["text"])
                if s.get("source_url"):
                    st.markdown(f"[Documento original]({s['source_url']})")

    if show_trace and data.get("trace"):
        st.subheader("Diálogo entre agentes")
        for m in data["trace"]:
            st.markdown(f"**{m['from']} → {m['to']}** · `{m['type']}`")
            st.json(m["content"], expanded=False)

    st.caption(data["disclaimer"])
