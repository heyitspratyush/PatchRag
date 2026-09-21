from pydantic import BaseModel




class Document(BaseModel):
    document_id : str
    document_text : str
   

class Chunk(BaseModel):
    chunk_id : str
    chunk_text : str
    document_id : str

    
class Evidence(BaseModel):
    evidence_id : str
    evidence_text : str
    chunk_id : str

class Claim(BaseModel):
    claim_id : str
    claim_text : str
    evidence_ids : list[str]

class Block(BaseModel):
    block_id : str
    block_text : str
    claim_ids : list[str]



class Intent(BaseModel):
    intent_id : str
    intent_text : str
    block_ids : list[str]



class Session(BaseModel):
    session_id : str

    intent_ids : list[str]