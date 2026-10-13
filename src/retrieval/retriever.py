from src.retrieval.embedder import embed_texts
from src.retrieval.index import search_index
from src.retrieval.index import map_indices_to_chunks
from src.state.models import Chunk

def retrieve_chunks(query: str, index, chunks: list, k: int):
    query_embeddings = embed_texts([query])

    query_vector = (query_embeddings.detach().cpu().numpy().astype("float32"))
    scores,indices = search_index(index, query_vector, k)

    retrieved_chunks = map_indices_to_chunks(indices, chunks)
    return retrieved_chunks, scores