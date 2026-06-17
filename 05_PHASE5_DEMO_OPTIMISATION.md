# Phase 5 — Optimisation & Interactive Demo

## Goal

Add multi-hop query decomposition, GDS-based re-ranking, and wrap the entire system in a polished Streamlit UI that visualises the retrieved subgraph alongside the generated answer.

---

## Prerequisites

- Phase 4 complete and evaluation metrics look reasonable
- Neo4j GDS plugin confirmed working (`CALL gds.version()` in Neo4j Browser)

---

## 5A — Query Decomposition (`src/query_decomposer.py`)

### Purpose

Complex biomedical questions often span multiple concepts. Decompose them into focused sub-questions, retrieve separately, and merge results. Example:

> "What is the relationship between BRCA1 mutations and chemotherapy response in triple-negative breast cancer?"
> → Sub-Q1: "BRCA1 mutations in breast cancer"
> → Sub-Q2: "chemotherapy response triple-negative breast cancer"
> → Sub-Q3: "BRCA1 chemotherapy sensitivity"

### Interface

```python
def decompose_query(query: str, llm_client) -> list[str]:
    """
    Uses LLM to decompose complex query into 2–4 sub-questions.
    Returns list of sub-question strings.
    Falls back to [query] if LLM returns parse error.
    """

def multi_hop_retrieve(query: str, pipeline, top_k: int = 5) -> list[dict]:
    """
    1. Decompose query into sub-questions
    2. retrieve() for each sub-question
    3. Merge all retrieved chunks
    4. Deduplicate by chunk_id
    5. Re-rank merged set by frequency (chunks appearing in multiple sub-retrievals rank higher)
    6. Return top_k * 2 chunks
    """
```

### Decomposition Prompt

```python
DECOMPOSE_PROMPT = """You are a biomedical research assistant. 
Break the following complex question into 2-4 simpler, focused sub-questions that together cover the original question.
Return ONLY a JSON array of strings. No explanation.

Question: {query}

Example output: ["sub-question 1", "sub-question 2", "sub-question 3"]"""
```

Parse the LLM response:
```python
import json, re

def parse_subquestions(response: str) -> list[str]:
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        return json.loads(match.group()) if match else []
    except Exception:
        return []
```

---

## 5B — GDS Re-ranking

### Purpose

Use PageRank on the retrieved subgraph to identify the most "central" chunks — those connected to many entities that are mentioned across multiple chunks. Central chunks are more likely to contain key information.

### Steps

```python
def gds_rerank(driver, chunk_ids: list[str]) -> dict[str, float]:
    """
    1. Project a named in-memory graph from the retrieved chunks + their entities
    2. Run PageRank on the projected graph
    3. Return {chunk_id: pagerank_score} mapping
    """
```

### Cypher: Project Subgraph

```cypher
CALL gds.graph.project.cypher(
  'retrieved_subgraph',
  'MATCH (n) WHERE (n:Chunk AND n.chunk_id IN $chunk_ids) OR
                    (n:Entity AND EXISTS {(c:Chunk)-[:MENTIONS]->(n) WHERE c.chunk_id IN $chunk_ids})
   RETURN id(n) AS id',
  'MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
   WHERE c.chunk_id IN $chunk_ids
   RETURN id(c) AS source, id(e) AS target',
  {parameters: {chunk_ids: $chunk_ids}}
)
YIELD graphName
```

### Cypher: Run PageRank

```cypher
CALL gds.pageRank.stream('retrieved_subgraph')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE node:Chunk
RETURN node.chunk_id AS chunk_id, score
ORDER BY score DESC
```

### Cleanup

```cypher
CALL gds.graph.drop('retrieved_subgraph', false)
```

### Combined Re-ranking Formula

```python
def combined_rerank(vector_results, gds_scores, alpha=0.6, beta=0.4):
    """
    final_score = alpha * vector_score + beta * normalized_gds_score
    """
```

---

## 5C — Streamlit Demo App (`app/demo.py`)

### App Layout

```
┌─────────────────────────────────────────────────────────┐
│  🧬 GraphRAG — Scientific Literature Q&A                │
├─────────────────────────────────────────────────────────┤
│  [🔍 Enter your biomedical question here...]  [Ask]     │
│                                                          │
│  ⚙️ Options:                                             │
│  [ ] Use graph expansion    [ ] Query decomposition      │
│  [ ] GDS re-ranking         Top-K: [5]                  │
├─────────────────────────────────────────────────────────┤
│  📝 Answer                                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LLM-generated answer shown here                  │  │
│  └───────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  🔗 Retrieved Subgraph                                   │
│  [Interactive pyvis graph — chunks & entities]           │
├─────────────────────────────────────────────────────────┤
│  📄 Source Chunks (expandable)                           │
│  Chunk 1: [article_id] [score] [text...]                 │
│  Chunk 2: ...                                            │
└─────────────────────────────────────────────────────────┘
```

### Key Streamlit Components

```python
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="GraphRAG — Scientific Literature", layout="wide")
st.title("🧬 GraphRAG — Scientific Literature Q&A")

# Input
query = st.text_input("Enter your biomedical question:", placeholder="e.g. What is the role of TNF-alpha in rheumatoid arthritis?")
col1, col2, col3 = st.columns(3)
with col1: use_graph = st.checkbox("Graph Expansion", value=True)
with col2: use_decomp = st.checkbox("Query Decomposition", value=False)
with col3: top_k = st.slider("Top-K", 3, 15, 5)

if st.button("Ask") and query:
    with st.spinner("Retrieving context..."):
        if use_decomp:
            chunks = multi_hop_retrieve(query, pipeline, top_k)
        else:
            chunks = pipeline.retrieve(query, top_k=top_k, expand_graph=use_graph)
    
    with st.spinner("Generating answer..."):
        answer = pipeline.generate(query, chunks)
    
    st.subheader("📝 Answer")
    st.write(answer)
    
    st.subheader("🔗 Retrieved Subgraph")
    render_graph(chunks)   # pyvis
    
    st.subheader("📄 Source Chunks")
    for c in chunks:
        with st.expander(f"[{c['article_id']}] Score: {c['score']:.3f}"):
            st.write(c['text'])
```

### Graph Visualisation with pyvis

```python
def render_graph(chunks: list[dict], driver) -> None:
    """Fetches chunk→entity connections for displayed chunks and renders with pyvis."""
    net = Network(height="400px", width="100%", bgcolor="#1a1a2e", font_color="white")
    
    chunk_ids = [c['chunk_id'] for c in chunks]
    
    # Fetch entities for displayed chunks
    with driver.session() as session:
        result = session.run("""
            UNWIND $ids AS cid
            MATCH (c:Chunk {chunk_id: cid})-[:MENTIONS]->(e:Entity)
            RETURN c.chunk_id AS chunk_id, e.name AS entity, e.type AS entity_type
        """, ids=chunk_ids)
        rows = list(result)
    
    # Add chunk nodes
    for c in chunks:
        net.add_node(c['chunk_id'], label=f"Chunk\n{c['chunk_id'][-6:]}", 
                     color="#4e9af1", shape="box", title=c['text'][:200])
    
    # Add entity nodes + edges
    added_entities = set()
    for row in rows:
        ent_id = f"entity_{row['entity']}"
        if ent_id not in added_entities:
            net.add_node(ent_id, label=row['entity'], color="#f1c40f", 
                        shape="ellipse", title=f"Type: {row['entity_type']}")
            added_entities.add(ent_id)
        net.add_edge(row['chunk_id'], ent_id)
    
    html = net.generate_html()
    components.html(html, height=420, scrolling=False)
```

---

## README.md Template

```markdown
# GraphRAG for Scientific Literature

A production-quality RAG system combining semantic chunking, a Neo4j knowledge graph, and graph-enhanced retrieval to answer complex biomedical questions over PubMed abstracts.

## Architecture
[diagram or ASCII art]

## Key Design Decisions
1. **Semantic Chunking over Fixed-Token**: HDBSCAN clusters respect topic boundaries...
2. **Graph Expansion**: Traversing Chunk→Entity→Chunk surfaces related abstracts...
3. **Combined Re-ranking**: α·vector_score + β·gds_pagerank balances relevance and centrality...

## Results
| Variant | Recall@10 | MRR | ROUGE-L | BERTScore |
|---|---|---|---|---|
| Baseline | ... | ... | ... | ... |
| Full System | ... | ... | ... | ... |

## Quick Start
\`\`\`bash
git clone ...
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
python src/chunker.py
python src/graph_builder.py
streamlit run app/demo.py
\`\`\`
```

---

## Deliverables Checklist

- [ ] `src/query_decomposer.py` with `decompose_query` and `multi_hop_retrieve`
- [ ] GDS PageRank re-ranking integrated into `rag_pipeline.py`
- [ ] `app/demo.py` — working Streamlit app with pyvis graph rendering
- [ ] `README.md` with architecture, results table, and quickstart
- [ ] GitHub repo with clean commit history and no secrets in code

---

## Dependencies for This Phase

```
streamlit
pyvis
networkx    # optional, fallback for graph viz
```
