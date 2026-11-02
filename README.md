# PatchRAG
### Dependency-Aware Selective Refinement for Retrieval-Augmented Generation

PatchRAG is a research-oriented Retrieval-Augmented Generation (RAG) system that investigates how an AI system should respond when a user changes or narrows a requirement after receiving an evidence-grounded answer.

Instead of restarting the entire RAG pipeline whenever a late constraint arrives, PatchRAG aims to identify exactly which parts of the previous answer are affected, preserve unaffected content, selectively retrieve replacement evidence, and verify that revised claims remain supported by their citations.

---

## 1. Research Question

> Can dependency-aware selective refinement after a late user constraint preserve answer quality and claim-level citation support while reducing latency, retrieval operations and generation tokens compared with full restart?

The research focuses on improving the efficiency and reliability of conversational RAG systems when users introduce new constraints after an answer has already been generated.

### Research Hypotheses

- H1: Selective refinement reduces retrieval operations and retrieved chunks compared with full restart.
- H2: Selective refinement reduces generation tokens and/or post-supplement latency.
- H3: Citation support does not degrade when changed claims are revalidated.
- H4: Naive patching can introduce citation orphaning.
- H5: Unaffected intents can preserve their correctness when a constraint targets only one intent.

These are research hypotheses, not assumed results.

---

## 2. Problem Statement

A conventional RAG pipeline follows:

User Query → Document Retrieval → Evidence → LLM Generation → Answer

However, consider the following interaction.

**Initial Query:**

"Explain blockchain and compare Proof of Work with Proof of Stake."

The system retrieves relevant documents and generates an answer.

**Late User Input:**

"For the comparison, focus on energy consumption and keep the blockchain explanation unchanged."

A conventional RAG system may retrieve documents and regenerate the entire answer.

This creates unnecessary computation because the blockchain explanation does not need to change.

A naive patching system introduces another problem: it may modify the answer while retaining citations that no longer support the revised claims.

### PatchRAG's Approach

PatchRAG investigates whether the system can:

1. Detect what changed in the user's request.
2. Identify the affected intent.
3. Identify affected answer blocks and claims.
4. Determine which evidence remains valid.
5. Preserve unaffected content.
6. Retrieve replacement evidence selectively.
7. Regenerate only affected answer blocks.
8. Verify the revised claims against their citations.

The objective is to minimize unnecessary recomputation without sacrificing answer quality or citation reliability.

---

## 3. Core Architecture

PatchRAG uses a dependency-aware architecture.

### Runtime Dependency Hierarchy

Session
  └── Intent
       └── Answer Block
            └── Claim
                 └── Evidence
                      └── Chunk
                           └── Document

| Component | Responsibility |
|---|---|
| Session | Maintains one conversation's state |
| Intent | Represents an individual user requirement |
| Answer Block | Contains the response associated with an intent |
| Claim | Represents an individual factual statement |
| Evidence | Information supporting a claim |
| Chunk | Retrieved segment of a document |
| Document | Original source document |

This hierarchy allows PatchRAG to identify which dependencies become invalid when a late constraint arrives.

### System Components

1. **Corpus and Retrieval**
   - Document ingestion
   - Text chunking
   - Embedding generation
   - Vector similarity search
   - Evidence retrieval

2. **Streaming Simulator**
   - Progressive transcript generation
   - Early retrieval simulation
   - Retrieval timing and cache telemetry

3. **Intent Splitter**
   - Decomposes requests into approximately 1–3 intents.
   - Maintains independent intent representations.

4. **Session State**
   - Stores intents, answer blocks, claims, evidence and citations.
   - Maintains relationships across conversational turns.

5. **Constraint Detector**
   - Classifies late user input.
   - Extracts constraint information.
   - Assigns classification confidence.

6. **Impact Mapper**
   - Identifies affected intents and claims.
   - Determines what should be preserved or recomputed.

7. **Selective Refinement**
   - Marks stale dependencies.
   - Reuses valid evidence.
   - Retrieves replacement evidence.
   - Regenerates affected answer blocks.

8. **Citation Verification**
   - Checks claim-evidence support.
   - Detects citation orphaning.
   - Revalidates the final answer.

9. **Telemetry and Evaluation**
   - Records quality metrics.
   - Measures retrieval operations.
   - Tracks latency and token consumption.
   - Stores experiment traces.

---

## 4. PatchRAG Workflow

### Initial Answer Generation

User Query
    ↓
Intent Decomposition
    ↓
Document Retrieval
    ↓
Evidence Collection
    ↓
Answer Generation
    ↓
Claim Extraction
    ↓
Citation Assignment
    ↓
Session State Storage

### Late Constraint Processing

Late User Input
    ↓
Constraint Classification
    ↓
Affected Intent Identification
    ↓
Affected Claim Identification
    ↓
Evidence Dependency Analysis
    ↓
Stale Dependency Detection
    ↓
Selective Evidence Retrieval
    ↓
Affected Block Regeneration
    ↓
Citation Verification
    ↓
Final Answer

Unaffected answer blocks and valid evidence are preserved.

### Late Input Categories

The constraint detector distinguishes:

- CONSTRAINT: Changes the scope or requirements of existing work.
- CLARIFICATION: Clarifies an existing intent.
- NEW_QUESTION: Introduces an independent question.
- NO_ACTION: Does not require answer refinement.
- UNCERTAIN: Cannot be classified confidently.

Low-confidence decisions enter a cautious refinement path.

---

## 5. Experimental Baselines

PatchRAG uses four configurations to evaluate the proposed refinement mechanism.

| ID | Configuration | Behavior |
|---|---|---|
| B0 | Ignore | Keeps the previous answer and ignores the late input |
| B1 | Full Restart | Reprocesses the complete updated query |
| B2 | Naive Patch | Performs selective refinement without citation verification |
| B3 | PatchRAG | Performs selective refinement with citation verification |

### Important Experimental Design Decision

B2 and B3 share the same selective retrieval and patching pipeline.

The primary difference is citation verification:

- B2: Verification OFF.
- B3: Verification ON.

This allows a fair comparison of citation safety without implementing two separate refinement systems.

---

## 6. Citation Verification and Citation Orphaning

One important problem in conversational RAG is citation orphaning.

Citation orphaning occurs when a claim is modified but its previous citation no longer supports the revised statement.

Example:

Original claim:
"Proof of Work requires substantial computational resources."

Original evidence supports this claim.

After a late constraint, the system generates a more specific claim about comparative energy consumption but accidentally retains the old citation.

The citation may no longer support the revised claim.

PatchRAG addresses this through claim-level verification.

Conceptually:

Claim + Cited Evidence
        ↓
   Verification
        ↓
SUPPORTED / UNSUPPORTED

The system should maintain traceability:

Final Claim → Evidence ID → Chunk ID → Document ID → Verification Result

Claims also have an explicit lifecycle:

- ACTIVE
- STALE
- SUPERSEDED

This allows refinement to track which claims remain valid and which have been replaced.

---

## 7. Benchmark and Evaluation

PatchRAG uses a controlled paired-conversation benchmark.

**Current documented benchmark size: 20 cases.**

Location:

`data/benchmark/benchmark.jsonl`

Each benchmark case contains:

- case_id
- session_id
- initial_query
- late_user_input
- expected_intents
- expected_constraint_type
- expected_affected_intents
- expected_affected_claims
- expected_unaffected_intents

### Benchmark Coverage

The benchmark includes:

- CONSTRAINT
- CLARIFICATION
- NEW_QUESTION
- NO_ACTION
- UNCERTAIN

It contains both single-intent and multi-intent scenarios.

### Ground Truth Isolation

The expected_* fields are evaluation ground truth only.

They must not be passed into the PatchRAG runtime as hints.

The system must independently discover affected intents, claims and evidence dependencies.

Runtime-generated claim IDs must also not be compared directly with benchmark claim IDs. Evaluation requires a content-level correspondence method.

Claim correspondence and citation support are separate measurements.

### Evaluation Metrics

**Answer Quality**
- Answer correctness
- Citation support rate
- Unaffected-content preservation

**Citation Safety**
- Citation orphaning rate
- Claim-to-evidence traceability
- Verifier-result coverage

**Efficiency**
- Retrieval calls
- Retrieved chunks
- Prompt tokens
- Generation tokens
- Post-supplement latency

**Streaming and Reuse**
- Time to first retrieval
- Cache hits

**Operational**
- Estimated cost
- Refinement decision measurements

### Planned Experiments

1. Full Restart vs PatchRAG.
2. Naive Patch vs PatchRAG.
3. Unaffected-content preservation.
4. Single-intent vs multi-intent constraints.
5. Different retrieval budgets.
6. Ablation of dependency mapping.
7. Ablation of evidence reuse.
8. Ablation of citation revalidation.

Experimental results must come from actual runs. No performance numbers should be invented or assumed.

---

## 8. Repository Structure

```text
PatchRAG/
│
├── data/
│   ├── corpus/
│   └── benchmark/
│       └── benchmark.jsonl
│
├── src/
│   │
│   ├── retrieval/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── index.py
│   │   └── retriever.py
│   │
│   ├── streaming/
│   │   ├── simulator.py
│   │   └── controller.py
│   │
│   ├── intent/
│   │   ├── splitter.py
│   │   └── constraint_detector.py
│   │
│   ├── state/
│   │   ├── models.py
│   │   └── session.py
│   │
│   ├── refinement/
│   │   ├── impact_mapper.py
│   │   ├── refiner.py
│   │   └── patcher.py
│   │
│   ├── verification/
│   │   ├── verifier.py
│   │   └── orphan_detector.py
│   │
│   ├── evaluation/
│   │   ├── benchmark.py
│   │   ├── metrics.py
│   │   └── runner.py
│   │
│   └── telemetry/
│       └── logger.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_state.py
│   └── test_benchmark.py
│
├── logs/
├── experiments/
├── demo/
├── docs/
│
├── .env.example
├── .gitignore
└── README.md
