import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "articles.json"

def load_articles():
    with open(DATA_PATH) as f:
        return json.load(f)