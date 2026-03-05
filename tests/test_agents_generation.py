from types import SimpleNamespace

from docint.agents.generation import ResultValidationResponseAgent
from docint.agents.types import RetrievalResult, Turn


class _FakeLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0

    def complete(self, prompt: str) -> SimpleNamespace:
        self.calls += 1
        _ = prompt
        return SimpleNamespace(text=self.response_text)


def test_validation_agent_sets_alert_on_mismatch() -> None:
    llm = _FakeLLM(
        '{"summary_grounded": false, "sources_relevant": true, "reason":"hallucinated fact"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert llm.calls == 1
    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is True
    assert finalized.validation_reason == "hallucinated fact"


def test_validation_agent_disabled_is_noop() -> None:
    llm = _FakeLLM(
        '{"summary_grounded": false, "sources_relevant": false, "reason":"bad"}'
    )
    agent = ResultValidationResponseAgent(enabled=False, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert llm.calls == 0
    assert finalized.validation_checked is None
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason is None
