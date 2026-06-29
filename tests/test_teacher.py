from tacet.core.graph import WorldGraph
from tacet.llm.teacher import CallableTeacher, Narrator, OracleTeacher


def test_oracle_teacher_perfect():
    def oracle(h, r):
        if h == "Alice" and r == "lives_in":
            return ["London"]
        return []

    teacher = OracleTeacher(oracle=oracle)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["London"]
    assert resp.cost == 1.0
    assert resp.correct is True


def test_oracle_teacher_noisy():
    def oracle(h, r):
        return ["London"]

    # error_rate=1.0 means it should always pick from the pool
    teacher = OracleTeacher(oracle=oracle, error_rate=1.0, entity_pool=["Paris"], seed=42)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["Paris"]
    assert resp.correct is False


def test_oracle_teacher_noisy_correct_by_chance():
    def oracle(h, r):
        return ["London"]

    # If the randomly chosen entity happens to be in the truth, correct should be True
    teacher = OracleTeacher(oracle=oracle, error_rate=1.0, entity_pool=["London"], seed=42)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["London"]
    assert resp.correct is True


def test_oracle_teacher_reproducibility():
    def oracle(h, r):
        return ["London"]

    pool = ["Paris", "Berlin", "Tokyo", "New York"]
    # With a fixed seed and error_rate, it should be deterministic
    t1 = OracleTeacher(oracle=oracle, error_rate=0.5, entity_pool=pool, seed=123)
    t2 = OracleTeacher(oracle=oracle, error_rate=0.5, entity_pool=pool, seed=123)

    graph = WorldGraph()
    results1 = [t1.answer(graph, "A", "R").answers for _ in range(10)]
    results2 = [t2.answer(graph, "A", "R").answers for _ in range(10)]

    assert results1 == results2


def test_callable_teacher():
    def my_llm(h, r):
        return ["Response from LLM"]

    teacher = CallableTeacher(fn=my_llm, cost=5.0)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["Response from LLM"]
    assert resp.cost == 5.0
    assert resp.correct is True


def test_narrator_render():
    narrator = Narrator()

    # Test multiple answers
    res1 = narrator.render("Alice", "lives_in", ["London", "Paris"], tier=3)
    assert "Alice — lives in → London, Paris" in res1
    assert "[via expert reasoning]" in res1

    # Test no answers
    res2 = narrator.render("Alice", "lives_in", [], tier=2)
    assert "No answer found for (Alice, lives in)." in res2

    # Test tier 1 with proof
    proof = ["Step 1: ...", "Step 2: ..."]
    res3 = narrator.render("Alice", "lives_in", ["London"], tier=1, proof=proof)
    assert "Alice — lives in → London" in res3
    assert "[via verified rules]" in res3
    assert "proof:" in res3
    assert "Step 1:" in res3
    assert "Step 2:" in res3
