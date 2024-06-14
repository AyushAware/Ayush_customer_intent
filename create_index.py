import sys
import json
import nltk
import chromadb
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def index_create(transcription_path, path, name):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index_path = f"{path}/{name}_index"
    chroma_client = chromadb.PersistentClient(path=index_path)

    with open(transcription_path, 'r') as file:
        data = json.load(file)
    
    collection = chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    doc = []

    for grp in data['groups']:
        text = grp['text']
        sentences = sent_tokenize(text)
        embeddings = model.encode(sentences)
        
        for s, e in zip(sentences, embeddings):
            doc_id = grp['id']
            doc.append({
                "ids": doc_id,
                "embeddings": e.tolist(),
                "metadatas": {
                    "department": grp['department'],
                    "text": s
                }
            })
    
    for d in doc:
        collection.add(
            ids=[d["ids"]],
            embeddings=[d["embeddings"]],
            metadatas=[d["metadatas"]]
        )

if __name__ == "__main__":
    index_create(
        r'C:\Users\Navya\Desktop\ayush_project\groups_final.json',
        r'C:\Users\Navya\Desktop\ayush_project\indexes',
        'collection_my'
    )
