# app.py
import streamlit as st
from infra.bootstrap import get_rag
from src.data_loader import load_articles


def main(articles):
    st.set_page_config(
        page_title="Interior Design Article Search",
        page_icon="🏠",
        layout="wide"
    )

    st.title("🏠 Architectural Articles Search")
    st.markdown(
        "*Lets design your dream house together*"
    )

    # Initialize RAG
    with st.spinner("Initializing system..."):
        rag = get_rag(articles)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Search Settings")

        num_results = st.slider("Number of results", 1, 10, 5)
        search_weight = st.slider(
            "Semantic vs Keyword weight",
            0.0, 1.0, 0.5
        )

        st.markdown("---")
        st.markdown("""
        **Hybrid RAG stack**
        - Dense retrieval (embeddings)
        - Sparse retrieval (keywords)
        - LLM summaries (OpenAI)
        """)

    # Search box
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search articles",
            placeholder="pink dining rooms, modern kitchens...",
            label_visibility="collapsed"
        )

    with col2:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    # Search execution
    if query and (search_clicked or query):
        with st.spinner("Searching and summarizing..."):
            results = rag.hybrid_search(
                query=query,
                top_k=num_results,
                alpha=search_weight
            )

        if not results:
            st.warning("No articles found.")
            return

        st.markdown(f"### Found {len(results)} articles")

        for idx, article in enumerate(results, 1):
            # Title as hyperlink
            article_url = article.get('url', '#')
            st.markdown(f"#### {idx}. [{article['title']}]({article_url})")

            with st.spinner("Generating summary..."):
                summary = rag.generate_summary(article)

            st.markdown(f"**Summary:** {summary}")

            # Show extracted keywords instead of tags
            if article.get("keywords"):
                keywords_html = " ".join(
                    f"<span style='background:#3f49cf;padding:2px 8px;border-radius:4px;margin-right:4px'>{k}</span>"
                    for k in article["keywords"]
                )
                st.markdown(f"**Keywords:** {keywords_html}", unsafe_allow_html=True)

            st.markdown("---")


if __name__ == "__main__":
    articles = load_articles()
    main(articles)
