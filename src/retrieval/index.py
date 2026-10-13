import faiss
import numpy as np
from src.state.models import Chunk


def build_index(embeddings):
    embeddings_numpy = (
        embeddings
        
        .cpu()
        .numpy()
        .astype("float32")
    )

    embedding_dimension = embeddings_numpy.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings_numpy)

    return index

def search_index(index,query_vector,k):
    query_vector = np.asarray(query_vector,dtype=np.float32)

    scores,indices= index.search(query_vector,k)
    return scores,indices

def map_indices_to_chunks(indices,chunks):
    retrieved_chunks=[]
    for chunk_index in indices[0]:
        if chunk_index<0:
            continue
        
        if chunk_index>=len(chunks):
            raise ValueError(f"Chunk index {chunk_index} is out of bounds for chunks list of length {len(chunks)}")
        
        retrieved_chunks.append(chunks[chunk_index])
    return retrieved_chunks