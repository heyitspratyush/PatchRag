import hashlib
from src.state.models import Chunk


def create_chunk_id(document_id : str , chunk_idx : int) -> str:
    raw_id = f"{document_id}:{chunk_idx}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[ :12]



def chunk_text(document_text:str,document_id:str)->list[Chunk]:
    words = document_text.split()
    chunk_size = 5
    overlap = 2
    start = 0
    end = min(start+chunk_size,len(words))
    chunks = []
    chunk_idx = 0
    while start<=len(words):
        chunk = words[start:min(end,len(words))]
        chunk_text = " ".join(chunk)
        chunk_id = create_chunk_id(document_id,chunk_idx)
        chunks.append(Chunk(chunk_id=chunk_id,chunk_text=chunk_text,document_id=document_id))
        chunk_idx+=1
        start = end-overlap
        end = start+chunk_size
        if end>len(words):
            end = len(words)+chunk_size+1
    return chunks
