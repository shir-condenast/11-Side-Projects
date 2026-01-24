from infra.state import get_state
from src.rag_loader import HybridRAG


def get_rag(articles):
    state = get_state()

    if not state.exists("rag"):
        rag = HybridRAG(articles)
        rag.populate_database()
        state.set("rag", rag)

    return state.get("rag")
