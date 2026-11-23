"""Genera el informe técnico de HealthRAG (HTML + PDF) a partir de los resultados.

    python docs/informe/build_informe.py                  # usa las carpetas por defecto
    python docs/informe/build_informe.py --exp3 reports_llm_exp3_v2

Lee los CSV de evaluación, arma las tablas y los gráficos, completa
`plantilla.html` y exporta `docs/Informe_HealthRAG.pdf` con Microsoft Edge o
Google Chrome en modo headless (si no encuentra ninguno, deja solo el HTML).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from string import Template

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figuras"
EXPS = ["EXP1", "EXP2", "EXP3"]
EXP_LABEL = {"EXP1": "EXP1 · básico", "EXP2": "EXP2 · avanzado", "EXP3": "EXP3 · multiagente"}

# Paleta de referencia (modo claro): slots categóricos en orden fijo + tinta de texto.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- formato
def num(x: float, d: int = 2) -> str:
    """Número con coma decimal (formato del informe)."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def secs(ms: float) -> str:
    return f"{num(ms, 1)} ms" if ms < 1000 else f"{num(ms / 1000, 1)} s"


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


# ----------------------------------------------------------------------------- datos
def e2e_stats(per_query: pd.DataFrame) -> pd.DataFrame:
    """Resumen por experimento a partir de las métricas por consulta."""
    d = per_query[per_query["exp"].isin(EXPS)].copy()
    d["abstained"] = d["abstained"].astype(bool)
    g = d.groupby("exp").agg(
        n=("query_id", "size"), faith=("faithfulness", "mean"),
        cit=("citation_precision", "mean"), rel=("cited_source_relevance", "mean"),
        ctx=("context_hit", "mean"), rouge=("rougeL", "mean"), bleu=("bleu", "mean"),
        ssim=("semantic_sim", "mean"), tokens=("tokens", "mean"), iters=("iterations", "mean"),
        p50=("latency_ms", "median"), p95=("latency_ms", lambda s: s.quantile(.95)),
        cost=("cost_usd", "mean"))
    sin = d[d["category"] == "sin_evidencia"]
    dire = d[d["category"] == "directa"]
    g["abst_ok"] = sin.groupby("exp")["abstained"].sum()
    g["n_sin"] = sin.groupby("exp").size()
    g["abst_falsa"] = dire.groupby("exp")["abstained"].sum()
    g["n_dir"] = dire.groupby("exp").size()
    g["llm_used"] = d.groupby("exp")["tokens"].apply(lambda s: int((s > 0).sum()))
    return g.fillna(0)


def load_llm(default_dir: str, exp3_dir: str | None) -> tuple[pd.DataFrame, str]:
    """Une EXP1-EXP2 de la corrida completa con EXP3 de su propia corrida (si existe).

    Si EXP3 no se repitió, usa solo las consultas de la corrida original anteriores
    al límite diario de Groq (Q001-Q014) y lo deja indicado en el informe."""
    base = pd.read_csv(ROOT / default_dir / "e2e_per_query.csv")
    if exp3_dir and (ROOT / exp3_dir / "e2e_per_query.csv").exists():
        e3 = pd.read_csv(ROOT / exp3_dir / "e2e_per_query.csv")
        e3 = e3[e3["exp"] == "EXP3"]
        merged = pd.concat([base[base["exp"].isin(["EXP1", "EXP2"])], e3], ignore_index=True)
        return merged, (f"EXP1 y EXP2 de <code>{default_dir}/</code>; EXP3 repetido aparte "
                        f"en <code>{exp3_dir}/</code> (29 consultas).")
    part = base[~((base["exp"] == "EXP3") & (base["query_id"] >= "Q015"))]
    return part, (f"EXP1 y EXP2 de <code>{default_dir}/</code> (29 consultas). "
                  "<strong>EXP3 parcial:</strong> solo Q001–Q014 (14 consultas directas), porque el "
                  "analista agotó el cupo diario de Groq a mitad del experimento; sin consultas "
                  "fuera del corpus en esta muestra.")


# ----------------------------------------------------------------------------- gráficos
def _style(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: num(v, 1)))


def _bars(ax, groups: list[str], series: dict[str, list[float]], colors: list[str],
          fmt=lambda v: num(v, 2), labels: bool = True, fontsize: float = 7.5) -> None:
    n = len(series)
    width = 0.8 / n
    for i, ((name, vals), col) in enumerate(zip(series.items(), colors)):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, vals, width * 0.9, color=col, label=name, edgecolor="white", linewidth=1)
        if not labels:
            continue
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, fmt(v), ha="center",
                    va="bottom", fontsize=fontsize, color=INK2)
    ax.set_xticks(range(len(groups)), groups)


def fig_recuperacion(rs: pd.DataFrame) -> str:
    order = ["basic", "hybrid", "advanced"]
    rs = rs.set_index("retrieval_mode").reindex([m for m in order if m in rs["retrieval_mode"].values])
    metrics = {"Hit": "hit@5", "Recall": "recall@5", "MRR": "mrr", "nDCG": "ndcg@5"}
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    names = {"basic": "Vectorial (básico)", "hybrid": "Híbrido BM25+RRF", "advanced": "Híbrido + reranker"}
    _bars(ax, list(metrics), {names[m]: [rs.loc[m, c] for c in metrics.values()] for m in rs.index},
          [C1, C2, C3])
    _style(ax)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15),
              labelcolor=INK)
    fig.tight_layout()
    out = FIG / "recuperacion.png"
    fig.savefig(out)
    plt.close(fig)
    return out.relative_to(HERE).as_posix()


def fig_calidad(ext: pd.DataFrame, llm: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), dpi=200, sharey=True)
    for ax, (title, col) in zip(axes, [("ROUGE-L", "rouge"), ("Precisión de citas", "cit"),
                                       ("Relevancia de fuentes", "rel")]):
        _bars(ax, ["EXP1", "EXP2", "EXP3"],
              {"Sin LLM (extractivo)": [ext.loc[e, col] for e in EXPS],
               "Con LLM (Groq)": [llm.loc[e, col] if e in llm.index else 0 for e in EXPS]},
              [C1, C2], fmt=lambda v: num(v, 2), fontsize=6.3)
        _style(ax)
        ax.set_title(title, fontsize=9.5, color=INK, pad=6)
        ax.set_ylim(0, 1.12)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIG / "calidad_respuestas.png"
    fig.savefig(out)
    plt.close(fig)
    return out.relative_to(HERE).as_posix()


# ----------------------------------------------------------------------------- grafo
# Posición (x, y, ancho, alto), título y subtítulo de cada nodo del StateGraph.
GRAFO_NODOS = {
    "__start__": (20, 64, 90, 48, "inicio", "pregunta filtrada"),
    "retrieve": (160, 60, 150, 56, "Recuperador", "retrieve"),
    "analyst": (380, 60, 150, 56, "Analista", "analyst · gpt-oss-120b"),
    "verifier": (610, 60, 150, 56, "Verificador", "verifier · gpt-oss-20b"),
    "finalize": (380, 212, 150, 56, "Salida", "finalize · citas o abstención"),
    "__end__": (190, 216, 90, 48, "fin", "respuesta + traza"),
}
# Trazo y etiqueta de cada arista: (path, [(x, y, texto, anchor)]).
GRAFO_ARISTAS = {
    ("__start__", "retrieve"): ("M110 88 H158", []),
    ("retrieve", "analyst"): ("M310 88 H378", [(344, 80, "evidencia", "middle")]),
    ("analyst", "verifier"): ("M530 78 H608", [(570, 70, "borrador", "middle")]),
    ("verifier", "analyst"): ("M610 100 H532", [(570, 116, "revise", "middle")]),
    ("verifier", "retrieve"): ("M685 60 C685 8, 235 8, 235 58",
                               [(460, 44, "retrieve_more: nueva búsqueda con la consulta sugerida", "middle")]),
    ("analyst", "finalize"): ("M455 116 V210", [(447, 160, "sin verificador", "end"),
                                                (447, 173, "(EXP1 y EXP2)", "end")]),
    ("verifier", "finalize"): ("M685 116 C685 200, 620 240, 532 240",
                               [(694, 170, "approve", "start"), (694, 183, "finalize_partial", "start"),
                                (694, 196, "abstain", "start")]),
    ("finalize", "__end__"): ("M378 240 H282", []),
}


def fig_grafo() -> str:
    """SVG del grafo de agentes armado a partir del StateGraph compilado.

    Los nodos y aristas salen de ``build_graph(...).get_graph()``; si el grafo del código
    cambia y el dibujo no lo contempla, falla en vez de mostrar un diagrama desactualizado.
    También guarda la versión Mermaid que genera LangGraph en ``figuras/grafo_langgraph.mmd``."""
    import sys

    sys.path.insert(0, str(ROOT))
    from app.agents.graph import build_graph

    g = build_graph("multiagent", None, None, object(), {"agents": {}}).get_graph()
    (FIG / "grafo_langgraph.mmd").write_text(g.draw_mermaid(), encoding="utf-8")
    nodes, edges = set(g.nodes), {(e.source, e.target): e.conditional for e in g.edges}
    if nodes != set(GRAFO_NODOS) or set(edges) != set(GRAFO_ARISTAS):
        raise RuntimeError(f"El grafo cambió: nodos {sorted(nodes)}, aristas {sorted(edges)}. "
                           "Actualizá GRAFO_NODOS y GRAFO_ARISTAS en build_informe.py.")
    parts = []
    for nid, (x, y, w, h, title, sub) in GRAFO_NODOS.items():
        terminal = nid.startswith("__")
        agent = nid in ("analyst", "verifier")
        cls = "gt" if terminal else ("ga" if agent else "gb")
        parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" '
                     f'rx="{h / 2 if terminal else 6}"/>')
        cx = x + w / 2
        parts.append(f'<text class="t" x="{cx}" y="{y + h / 2 - 3}" text-anchor="middle">{title}</text>')
        parts.append(f'<text class="s" x="{cx}" y="{y + h / 2 + 12}" text-anchor="middle">{sub}</text>')
    for key, (d, labels) in GRAFO_ARISTAS.items():
        parts.append(f'<path class="{"gc" if edges[key] else "gl"}" d="{d}"/>')
        parts += [f'<text class="s" x="{lx}" y="{ly}" text-anchor="{a}">{txt}</text>'
                  for lx, ly, txt, a in labels]
    return (
        '<svg viewBox="0 0 800 290" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Grafo de estados de LangGraph: recuperador, analista, verificador y salida">'
        '<defs><marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker>'
        '<style>.gb{fill:#fff;stroke:#52514e;stroke-width:1.2}.ga{fill:#e8f0fb;stroke:#2a78d6;'
        'stroke-width:1.5}.gt{fill:#f4f3f0;stroke:#7a7973;stroke-width:1.2}'
        '.t{font-size:12px;fill:#0b0b0b;font-weight:600}.s{font-size:10px;fill:#52514e}'
        '.gl{stroke:#52514e;stroke-width:1.3;fill:none;marker-end:url(#ag)}'
        '.gc{stroke:#52514e;stroke-width:1.2;fill:none;stroke-dasharray:4 3;marker-end:url(#ag)}'
        '</style></defs>' + "".join(parts) + "</svg>")


# ----------------------------------------------------------------------------- tablas
def t_recuperacion(rs: pd.DataFrame) -> str:
    names = {"basic": "Vectorial (básico)", "hybrid": "Híbrido BM25 + RRF", "advanced": "Híbrido + reranker"}
    rows = []
    for m in ["basic", "hybrid", "advanced"]:
        r = rs[rs["retrieval_mode"] == m]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append([names[m], secs(r["retrieval_ms"]), num(r["hit@5"], 3), num(r["recall@5"], 3),
                     num(r["precision@5"], 3), num(r["mrr"], 3), num(r["ndcg@5"], 3),
                     num(r["recall@10"], 3), num(r["useful_context@5"], 3)])
    return table(["Modo", "Tiempo", "Hit", "Recall", "Precision", "MRR", "nDCG",
                  "Recall (10 resultados)", "Contexto útil"], rows, "num")


def t_respuestas(g: pd.DataFrame, llm: bool) -> str:
    rows = []
    for e in EXPS:
        if e not in g.index:
            continue
        r = g.loc[e]
        abst_ok = f"{int(r.abst_ok)} de {int(r.n_sin)}" if r.n_sin else "—"
        row = [EXP_LABEL[e], str(int(r.n)), num(r.faith), num(r.cit), num(r.rel), num(r.rouge),
               num(r.bleu), num(r["ssim"]), abst_ok, f"{int(r.abst_falsa)} de {int(r.n_dir)}",
               secs(r.p50), secs(r.p95)]
        if llm:
            row += [num(r.tokens, 0), num(r.iters, 2)]
        rows.append(row)
    headers = ["Experimento", "n", "Fidelidad", "Precisión de citas", "Relevancia de fuentes",
               "ROUGE&#8209;L", "BLEU", "Similitud semántica", "Abstención correcta",
               "Falsa abstención", "Latencia p50", "Latencia p95"]
    if llm:
        headers += ["Tokens por consulta", "Iteraciones"]
    return table(headers, rows, "num small")


def t_exp4(summary: pd.DataFrame) -> str:
    s = summary.set_index("exp")
    rows = []
    for key, label in [("EXP4_cache_frio", "Caché vacía"), ("EXP4_cache_caliente", "Caché llena"),
                       ("EXP4_top3", "top-k = 3"), ("EXP3", "top-k = 5 (base)"), ("EXP4_top8", "top-k = 8")]:
        if key not in s.index:
            continue
        r = s.loc[key]
        rows.append([label, secs(r["latency_p50_ms"]), secs(r["latency_p95_ms"]),
                     num(r["context_hit"], 3), num(r["cited_source_relevance"], 3), num(r["rougeL"], 3)])
    return table(["Configuración (multiagente, sin LLM)", "Latencia p50", "Latencia p95",
                  "Contexto encontrado", "Relevancia de fuentes", "ROUGE-L"], rows, "num")


def t_groq10(summary: pd.DataFrame) -> str:
    s = summary.set_index("exp")
    rows = [[EXP_LABEL[e], num(s.loc[e, "rougeL"]), num(s.loc[e, "citation_precision"]),
             num(s.loc[e, "faithfulness"]), secs(s.loc[e, "latency_p50_ms"]), num(s.loc[e, "tokens"], 0)]
            for e in EXPS if e in s.index]
    return table(["Experimento", "ROUGE-L", "Precisión de citas", "Fidelidad", "Latencia p50",
                  "Tokens por consulta"], rows, "num")


# ----------------------------------------------------------------------------- PDF
def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def to_pdf(html: Path, pdf: Path) -> bool:
    browser = find_browser()
    if not browser:
        print("No se encontró Edge/Chrome: abrí el HTML y exportalo a PDF desde el navegador.")
        return False
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return pdf.exists()


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extractivo", default="reports_extractivo")
    ap.add_argument("--groq10", default="reports_groq")
    ap.add_argument("--llm", default="reports_llm_v2", help="Corrida completa con LLM (EXP1-EXP3)")
    ap.add_argument("--exp3", default="reports_llm_exp3_v2", help="Corrida aparte de EXP3, si existe")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)

    rs = pd.read_csv(ROOT / args.extractivo / "retrieval_summary.csv")
    ext_pq = pd.read_csv(ROOT / args.extractivo / "e2e_per_query.csv")
    ext_sum = pd.read_csv(ROOT / args.extractivo / "e2e_summary.csv")
    g10 = pd.read_csv(ROOT / args.groq10 / "e2e_summary.csv")
    llm_pq, llm_note = load_llm(args.llm, args.exp3)
    ext, llm = e2e_stats(ext_pq), e2e_stats(llm_pq)

    values = {
        "tabla_recuperacion": t_recuperacion(rs),
        "tabla_extractivo": t_respuestas(ext, llm=False),
        "tabla_llm": t_respuestas(llm, llm=True),
        "tabla_exp4": t_exp4(ext_sum),
        "tabla_groq10": t_groq10(g10),
        "nota_llm": llm_note,
        "fig_recuperacion": fig_recuperacion(rs),
        "fig_calidad": fig_calidad(ext, llm),
        "fig_grafo": fig_grafo(),
        "llm_uso": ", ".join(f"{e}: {int(llm.loc[e, 'llm_used'])} de {int(llm.loc[e, 'n'])}"
                             for e in EXPS if e in llm.index),
        "rouge_ext": num(ext.loc["EXP1", "rouge"]), "rouge_llm": num(llm.loc["EXP1", "rouge"]),
        "rouge_ext2": num(ext.loc["EXP2", "rouge"]), "rouge_llm2": num(llm.loc["EXP2", "rouge"]),
        "cit_llm1": num(llm.loc["EXP1", "cit"]), "cit_llm2": num(llm.loc["EXP2", "cit"]),
        "cit_llm3": num(llm.loc["EXP3", "cit"]) if "EXP3" in llm.index else "—",
    }
    e3 = llm_pq[llm_pq["exp"] == "EXP3"]
    values["exp3_n"] = str(len(e3))
    values["exp3_iter2"] = str(int((e3["iterations"] > 1).sum()))
    values["exp3_ratio"] = num(e3["tokens"].mean() / llm.loc["EXP1", "tokens"], 1) if len(e3) else "—"
    html = Template((HERE / "plantilla.html").read_text(encoding="utf-8")).substitute(values)
    out_html = HERE / "informe.html"
    out_html.write_text(html, encoding="utf-8")
    print("HTML:", out_html)
    if not args.no_pdf:
        pdf = ROOT / "docs" / "Informe_HealthRAG.pdf"
        if to_pdf(out_html, pdf):
            print("PDF:", pdf)


if __name__ == "__main__":
    main()
