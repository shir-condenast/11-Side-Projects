# app.py
import streamlit as st
from infra.bootstrap import get_rag
from src.data_loader import load_articles


def main(articles):
    st.set_page_config(
        page_title="Interior Design Consultant",
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 Your Interior Design Consultant")
    st.markdown(
        "*Tell me what you're dreaming of, and I'll share some inspiring ideas*"
    )

    # Initialize RAG
    with st.spinner("Initializing your design assistant..."):
        rag = get_rag(articles)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Preferences")

        num_results = st.slider("How many ideas?", 1, 10, 5)
        search_weight = st.slider(
            "Style matching",
            0.0, 1.0, 0.5,
            help="0 = keyword match, 1 = style/mood match"
        )

        st.markdown("---")
        st.markdown("""
        **Powered by**
        - 🔍 Hybrid search (keyword + semantic)
        - 🎯 CrossEncoder reranking
        - 🤖 GPT-4o-mini recommendations
        """)

    # Chat-like input
    query = st.chat_input("What are you looking for? (e.g., 'pink dining rooms', 'cozy bedroom ideas')")

    if query:
        # User message
        with st.chat_message("user"):
            st.write(query)

        # Assistant response
        with st.chat_message("assistant", avatar="🎨"):
            with st.spinner("Let me think about this..."):
                results = rag.hybrid_search(
                    query=query,
                    top_k=num_results,
                    alpha=search_weight
                )

            # GUARDRAIL: Handle no relevant results gracefully
            if not results:
                no_results_msg = rag.generate_no_results_response(query)
                st.write(no_results_msg)
                st.markdown("")
                st.markdown("💡 **Try asking about:**")
                st.markdown("- Pink dining rooms, blush bedrooms")
                st.markdown("- Modern kitchens, cozy living spaces")
                st.markdown("- Color palettes (navy, emerald, coral)")
                return

            # Generate conversational intro
            with st.spinner(""):
                intro = rag.generate_conversation_intro(query, results)

            st.write(intro)
            st.markdown("")
            st.markdown("**Here are my recommendations:**")
            st.markdown("")

            # Display each recommendation
            for idx, article in enumerate(results, 1):
                article_url = article.get('url', '#')

                # Generate personalized recommendation
                with st.spinner(""):
                    recommendation = rag.generate_recommendation(article, query)

                # Article title as link
                st.markdown(f"**{idx}. [{article['title']}]({article_url})**")

                # Personalized recommendation text (grounded in article content)
                st.markdown(f"→ {recommendation}")

                # Keywords as subtle tags
                if article.get("keywords"):
                    keywords_html = " ".join(
                        f"<span style='background:#e8e8e8;color:#555;padding:2px 6px;border-radius:3px;font-size:0.8em;margin-right:3px'>{k}</span>"
                        for k in article["keywords"]
                    )
                    st.markdown(keywords_html, unsafe_allow_html=True)

                st.markdown("")


if __name__ == "__main__":
    articles = load_articles()
    main(articles)
