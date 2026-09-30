from src.state.models import ClaimStatus
from src.state.models import (
    Document,
    Chunk,
    Evidence,
    Claim,
    Block,
    Intent,
    Session,
)


def test_state_objects_can_be_linked():
    document = Document(
        document_id="doc_001",
        document_text="Solar energy information.",
    )

    chunk = Chunk(
        chunk_id="chunk_001",
        chunk_text="Solar energy comes from sunlight.",
        document_id=document.document_id,
    )

    evidence = Evidence(
        evidence_id="evidence_001",
        evidence_text="Solar energy comes from sunlight.",
        chunk_id=chunk.chunk_id,
    )

    claim = Claim(
        claim_id="claim_001",
        claim_text="Solar energy comes from sunlight.",
        evidence_ids=[evidence.evidence_id],
    )

    block = Block(
        block_id="block_001",
        block_text="Solar energy explanation.",
        claim_ids=[claim.claim_id],
    )

    intent = Intent(
        intent_id="intent_001",
        intent_text="Explain solar energy.",
        block_ids=[block.block_id],
    )

    session = Session(
        session_id="session_001",
        intent_ids=[intent.intent_id],
    )

    assert chunk.document_id == document.document_id
    assert evidence.chunk_id == chunk.chunk_id
    assert claim.evidence_ids == [evidence.evidence_id]
    assert block.claim_ids == [claim.claim_id]
    assert intent.block_ids == [block.block_id]
    assert session.intent_ids == [intent.intent_id]
    assert claim.status == ClaimStatus.ACTIVE