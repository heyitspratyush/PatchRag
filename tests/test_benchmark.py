from src.evaluation.benchmark import load_benchmark, ConstraintType


def test_benchmark_loads():
    cases = load_benchmark("data/benchmark/benchmark.jsonl")

    assert len(cases) == 20


def test_multi_intent_case():
    cases = load_benchmark("data/benchmark/benchmark.jsonl")

    case = cases[1]

    assert len(case.expected_intents) == 2
    assert case.expected_constraint_type == ConstraintType.CONSTRAINT
    assert case.expected_affected_intents == ["intent_002"]
    assert case.expected_unaffected_intents == ["intent_001"]