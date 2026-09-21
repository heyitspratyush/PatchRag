from enum import Enum
import json
from pathlib import Path

from pydantic import BaseModel


class ConstraintType(str, Enum):
    CONSTRAINT = "CONSTRAINT"
    CLARIFICATION = "CLARIFICATION"
    NEW_QUESTION = "NEW_QUESTION"
    NO_ACTION = "NO_ACTION"
    UNCERTAIN = "UNCERTAIN"

class ExpectedIntent(BaseModel):
    intent_id: str
    intent_text: str


class BenchmarkCase(BaseModel):
    case_id: str
    session_id: str
    initial_query: str
    
    late_user_input: str
    expected_intents: list[ExpectedIntent]
    expected_constraint_type: ConstraintType
    expected_affected_intents: list[str]
    expected_affected_claims: list[str]
    expected_unaffected_intents: list[str]


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    cases = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            data = json.loads(line)
            case = BenchmarkCase.model_validate(data)
            cases.append(case)

    return cases