import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

def index_query(query, index_path, collection_name, top_n=1):
    chroma_client = chromadb.PersistentClient(path=index_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(name=collection_name)
    
    emb_query = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=emb_query,
        n_results=top_n,
    )
    
    top_n_data = []
    for i, meta, dist in zip(results['ids'][0], results['metadatas'][0], results['distances'][0]):
        top_n_data.append({
            "department": meta["department"],
            "text": meta["text"]
        })
    
    return top_n_data

if __name__ == "__main__":
    transcription_path = r'"E:\ayush_project\indexes\collection_my_index"'
    collection_name = 'collection_my'
    # query = "Novo allowed our account to be hacked!!! Do not use Novo. More than $8000 was stolen from our business account. We reported this to Novo and no one from the fraud department or anywhere else has contacted us back to resolve the issue nor have we been offered the protected amount of $250,000 that our money will be protected, was not protected at all, this bank is the worst. Novoâ€™s agent is responsible for the hack. They allowed our business email address to be changed by a hacker and allowed our account to be broken into and no one is trying to fix it on Novo side."
    # top_n = 1
    csv_path=r'"E:\ayush_project\Training_data\Book4.csv"'

    df = pd.read_csv(csv_path)
    
    if 'Reviewed text' not in df.columns:
        raise ValueError("The CSV file must contain a 'Reviewed text' column")
    
    
    df['Department'] = np.nan
    df['Intent'] = np.nan
    
    for idx, row in df.iterrows():
        query = row['Reviewed text']
        if isinstance(query, str):  
            try:
                top_n_results = index_query(query, transcription_path, collection_name, top_n=1)
                if top_n_results:
                    result = top_n_results[0]
                    df.at[idx, 'Department'] = result['department']
                    df.at[idx, 'Intent'] = result['text']
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
    
    output_csv_path = r'"E:\ayush_project\res_book1.csv".csv'
    df.to_csv(output_csv_path, index=False)
    
    print(f"Results saved to {output_csv_path}")








    # top_n_results = index_query(query, transcription_path, collection_name, top_n)
    # for result in top_n_results:
    #     print(f"This is the intent: {result['text']} and this is the department: {result['department']}")
