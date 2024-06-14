import nltk
import chromadb
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import json
import uuid

with open(r'E:\ayush_intent_prediction\intent.json', 'r') as f:
    data=json.load(f)

result = {"groups": []}

for department, texts in data.items():
    for text in texts:
        result["groups"].append({
            "department": department,
            "text": text,
            "id": str(uuid.uuid4())
        })


file_path = r'E:\ayush_intent_prediction\groups.json'
with open(file_path, 'w') as file:
    json.dump(result, file, indent=4)
