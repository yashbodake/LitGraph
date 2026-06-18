"""Phase 5C — interactive Streamlit demo for the GraphRAG pipeline.

Run from the project root:

    streamlit run app/demo.py

The app wires up ``GraphRAGPipeline`` (driver + embedder + LLM) once and exposes
toggles for graph expansion, multi-hop query decomposition, and GDS PageRank
re-ranking. Retrieved chunks and their entities are visualised with pyvis.
"""

# 1. stdlib
import logging
import sys
from pathlib import Path

# 2. third-party
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# Make ``src`` importable when running ``streamlit run app/demo.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 3. local
from src.rag_pipeline import GraphRAGPipeline  # noqa: E402
from src.query_decomposer import multi_hop_retrieve  # noqa: E402

logger = logging.getLogger(__name__)

# Page config must be the first Streamlit command.
st.set_page_config(
    page_title="GraphRAG — Scientific Literature",
    page_icon=":dna:",
    layout="wide",
)
st.title("GraphRAG — Scientific Literature Q&A")
st.caption(
    "Semantic chunking + Neo4j knowledge-graph traversal + graph-enhanced "
    "generation over PubMed abstracts."
)


@st.cache_resource(show_spinner="Loading pipeline (embedder + Neo4j + LLM)...")
def get_pipeline() -> GraphRAGPipeline:
    """Build the pipeline once and cache it across reruns."""
    return GraphRAGPipeline()


@st.cache_data(show_spinner=False)
def fetch_chunk_entities(_driver, chunk_ids: tuple[str, ...]) -> list[dict]:
    """Return ``(chunk_id, entity, entity_type)`` rows for the given chunks.

    The leading underscore lets Streamlit cache a driver-dependent function by
    its other args (the hashable ``chunk_ids`` tuple).
    """
    if not chunk_ids:
        return []
    cypher = """
    UNWIND $ids AS cid
    MATCH (c:Chunk {chunk_id: cid})-[:MENTIONS]->(e:Entity)
    RETURN c.chunk_id AS chunk_id, e.name AS entity, e.type AS entity_type
    """
    with _driver.session() as session:
        result = session.run(cypher, ids=list(chunk_ids))
        return [dict(r) for r in result]


def render_graph(chunks: list[dict], driver) -> None:
    """Render the retrieved chunk–entity subgraph with pyvis."""
    net = Network(height="420px", width="100%", bgcolor="#1a1a2e", font_color="white")
    net.repulsion(node_distance=120, spring_length=110)

    for chunk in chunks:
        net.add_node(
            chunk["chunk_id"],
            label=f"Chunk\n{str(chunk['chunk_id'])[-6:]}",
            color="#4e9af1",
            shape="box",
            title=chunk.get("text", "")[:200],
        )

    try:
        rows = fetch_chunk_entities(driver, tuple(c["chunk_id"] for c in chunks))
    except Exception as exc:  # noqa: BLE001 — viz must not crash the app
        st.info(f"Could not load entities for the graph view ({exc}).")
        rows = []

    added_entities: set[str] = set()
    for row in rows:
        ent_id = f"entity_{row['entity']}"
        if ent_id not in added_entities:
            net.add_node(
                ent_id,
                label=row["entity"],
                color="#f1c40f",
                shape="ellipse",
                title=f"Type: {row.get('entity_type', 'unknown')}",
            )
            added_entities.add(ent_id)
        net.add_edge(row["chunk_id"], ent_id)

    try:
        html = net.generate_html()
        components.html(html, height=440, scrolling=False)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Graph rendering failed ({exc}).")


# --------------------------------------------------------------------------- #
# Sidebar options
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Options")
    use_graph = st.checkbox(
        "Graph expansion", value=True,
        help="Traverse Chunk -> Entity -> Chunk to surface related abstracts.",
    )
    use_decomp = st.checkbox(
        "Query decomposition", value=False,
        help="Split complex questions into sub-questions and merge results.",
    )
    use_gds = st.checkbox(
        "GDS re-ranking", value=False,
        help="Re-rank chunks by PageRank centrality over the retrieved subgraph.",
    )
    top_k = st.slider(
        "Top-K", 3, 15, 5,
        help="Number of seed chunks to retrieve per (sub-)question.",
    )
    st.divider()
    st.markdown("**Provider:** set `CEREBRAS_API_KEY` / `OPENAI_API_KEY` in `.env`, else Ollama.")


query = st.text_input(
    "Enter your biomedical question:",
    placeholder="e.g. What is the role of TNF-alpha in rheumatoid arthritis?",
)

if st.button("Ask", type="primary") and query.strip():
    try:
        pipeline = get_pipeline()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialise the pipeline: {exc}")
        st.stop()

    try:
        with st.spinner("Retrieving context..."):
            if use_decomp:
                chunks = multi_hop_retrieve(query, pipeline, top_k)
            elif use_gds:
                chunks = pipeline.retrieve_with_gds(
                    query, top_k=top_k, expand_graph=use_graph
                )
            else:
                chunks = pipeline.retrieve(
                    query, top_k=top_k, expand_graph=use_graph
                )

        if not chunks:
            st.warning("No chunks retrieved. Check that Neo4j is populated.")
            st.stop()

        with st.spinner("Generating answer..."):
            answer = pipeline.generate(query, chunks)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Pipeline error: {exc}")
        st.stop()

    st.subheader("Answer")
    st.write(answer)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Source chunks")
        for i, chunk in enumerate(chunks, 1):
            score = chunk.get("score", 0.0)
            extra = ""
            if "frequency" in chunk:
                extra = f" · found by {chunk['frequency']} sub-questions"
            with st.expander(
                f"[{i}] Article {chunk.get('article_id', '?')} · score {score:.3f}{extra}"
            ):
                st.write(chunk.get("text", ""))
    with right:
        st.subheader("Retrieved subgraph")
        render_graph(chunks, pipeline.driver)
else:
    st.info("Enter a question above and press **Ask**.")
