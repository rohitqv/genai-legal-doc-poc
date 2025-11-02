"""
Streamlit Dashboard for Legal Document Analysis
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.rag.embed_faiss import EmbeddingManager
from src.rag.query_langchain import RAGPipeline
from src.utils.config import FAISS_INDEX_PATH
from src.utils.logger import logger

# Page configuration
st.set_page_config(
    page_title="Legal Document Analysis",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False


def load_rag_pipeline():
    """Load RAG pipeline and FAISS index"""
    try:
        embedding_manager = EmbeddingManager()
        faiss_index, metadata = embedding_manager.load_index(FAISS_INDEX_PATH)
        
        rag_pipeline = RAGPipeline(
            embedding_manager=embedding_manager,
            faiss_index=faiss_index,
            metadata=metadata
        )
        
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.index_loaded = True
        return True
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {str(e)}")
        return False


# Sidebar
st.sidebar.title("📄 Legal Document Analysis")
st.sidebar.markdown("---")

# Load index button
if st.sidebar.button("Load RAG Pipeline"):
    with st.spinner("Loading RAG pipeline..."):
        if load_rag_pipeline():
            st.sidebar.success("RAG pipeline loaded successfully!")

if not st.session_state.index_loaded:
    st.warning("⚠️ Please load the RAG pipeline from the sidebar first.")
    st.stop()

# Main content
st.title("Legal Document Analysis Dashboard")
st.markdown("Query your legal documents using natural language")

# Tab selection
tab1, tab2, tab3 = st.tabs(["🔍 Natural Language Query", "📊 Extracted Attributes", "📈 Statistics"])

# Tab 1: Natural Language Query
with tab1:
    st.header("Ask Questions About Your Documents")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main financial risks mentioned in the documents?",
        key="query_input"
    )
    
    if st.button("Search", type="primary") or query:
        if query:
            with st.spinner("Searching documents..."):
                result = st.session_state.rag_pipeline.query(query)
                
                st.subheader("Answer")
                st.write(result["answer"])
                
                st.subheader(f"Retrieved {result['num_chunks']} Relevant Chunks")
                
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    with st.expander(f"Chunk {i} (Score: {chunk['score']:.4f})"):
                        st.write(f"**Document ID:** {chunk['doc_id']}")
                        st.write(f"**Chunk Index:** {chunk['chunk_index']}")
                        st.write("**Content:**")
                        st.write(chunk["text"])

# Tab 2: Extracted Attributes
with tab2:
    st.header("Browse Extracted Attributes")
    st.markdown("View extracted attributes from processed documents")
    
    # Example query to show extracted data
    # In production, this would query the Gold layer Delta table
    st.info("💡 This section would display attributes extracted and stored in the Gold layer Delta table.")
    
    attribute_filter = st.selectbox(
        "Filter by Attribute:",
        ["All", "Company Name", "Filing Date", "Risk Factors", "Revenue", "Compliance"]
    )
    
    if st.button("Load Extracted Attributes"):
        st.success("✅ Attributes loaded from Gold layer")
        st.dataframe({
            "Document ID": ["doc_001", "doc_002", "doc_003"],
            "Attribute": ["Risk Factors", "Revenue", "Compliance"],
            "Value": ["Market volatility, Regulatory changes", "$394.3B", "SEC compliant"],
            "Confidence": [0.95, 0.92, 0.88]
        })

# Tab 3: Statistics
with tab3:
    st.header("Pipeline Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", "150")
        st.caption("Documents in Bronze layer")
    
    with col2:
        st.metric("Total Chunks", "2,450")
        st.caption("Text chunks in Silver layer")
    
    with col3:
        st.metric("Extracted Attributes", "1,200")
        st.caption("Attributes in Gold layer")
    
    st.markdown("---")
    
    st.subheader("Processing Metrics")
    st.info("💡 In production, this would query the audit_log Delta table for real-time metrics.")
    
    # Example chart data
    import pandas as pd
    import plotly.express as px
    
    metrics_data = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "Documents Processed": [10, 15, 12, 18, 20, 15, 22, 18, 25, 20, 28, 22, 30, 25, 32, 28, 35, 30, 38, 32, 40, 35, 42, 38, 45, 40, 48, 42, 50, 45]
    })
    
    fig = px.line(metrics_data, x="Date", y="Documents Processed", title="Documents Processed Over Time")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Legal Document Analysis PoC** | Built with Databricks, LangChain, and Streamlit")

