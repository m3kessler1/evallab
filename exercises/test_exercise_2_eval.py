"""Exercise 2: Write DeepEval Assertions

In this exercise, you'll write automated evaluations using DeepEval's LLM-as-a-judge.
Unlike traditional assertions that check for exact string matches, these metrics
evaluate semantic meaning and factual correctness.

TODO:
1. Import DeepEval metrics (AnswerRelevancyMetric, FaithfulnessMetric)
2. Create test cases with input, actual_output, and retrieval_context
3. Set appropriate thresholds for each metric
4. Run evaluations using deepeval test run
5. Analyze the LLM judge's reasoning
"""

import pytest
import sys
from pathlib import Path

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
try:
    from src.agent import build_oak_brook_support_agent, _build_llm, _get_provider_config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.agent import build_oak_brook_support_agent, _build_llm, _get_provider_config

import os


class _ProjectLLM(DeepEvalBaseLLM):
    """DeepEval judge model backed by the project's .env LLM config."""

    def __init__(self):
        self._llm = _build_llm()
        config = _get_provider_config()
        self._model_name = os.getenv(config["model_env"], config["default_model"])

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return self._model_name


def _load_deepeval():
    """Import DeepEval lazily to avoid pre-import plugin rewrite warnings."""
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    return assert_test, LLMTestCase


def _build_oak_brook_support_agent():
    """Import and build the support agent lazily during test execution."""
    try:
        from src.agent import build_oak_brook_support_agent
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from src.agent import build_oak_brook_support_agent
    return build_oak_brook_support_agent()


_eval_model = _ProjectLLM()
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=_eval_model)
faithfulness = FaithfulnessMetric(threshold=1.0, model=_eval_model)


# Knowledge base context for faithfulness evaluation
KNOWLEDGE_BASE_CONTEXT = [
    "Healthcare IT Support: We ensure all HIPAA regulations regarding technology are followed.",
    "Cybersecurity Services: We provide advanced solutions including firewalls, threat detection, and vulnerability assessments.",
    "Cloud Services: We offer migration, management, and optimization of cloud infrastructure.",
    "Data Backup & Recovery: Automated daily backups with 99.9% recovery success rate.",
    "Network Infrastructure: Design, implementation, and 24/7 monitoring with 99.99% uptime SLA."
]


@pytest.mark.parametrize(
    "user_query,expected_topics",
    [
        ("What healthcare IT services do you offer?",
         ["HIPAA", "healthcare", "patient data"]),
        ("Tell me about cybersecurity", [
            "firewalls", "threat detection", "vulnerability"]),
        ("Do you do cloud migration?", [
            "cloud", "migration", "AWS", "Azure"]),
    ]
)
def test_oak_brook_support_agent_quality(user_query, expected_topics):
    """
    Test the agent's output quality using DeepEval metrics.

    This test evaluates:
    - Answer Relevancy: Does the answer address the user's question?
    - Faithfulness: Is the answer grounded in the knowledge base?
    """
    # Import DeepEval components
    assert_test, LLMTestCase = _load_deepeval()
    
    # Build and run the agent
    executor = build_oak_brook_support_agent()
    response = executor.invoke({"input": user_query})
    actual_output = response["output"]

    # Create the test case
    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_output,
        retrieval_context=KNOWLEDGE_BASE_CONTEXT
    )

    # Assert against the metrics
    assert_test(test_case, [answer_relevancy, faithfulness])


# Starter test without DeepEval (remove this when you implement above)
def test_agent_runs():
    """Basic smoke test to ensure agent executes without errors."""
    executor = _build_oak_brook_support_agent()
    test_query = "What healthcare IT services do you offer?"
    response = executor.invoke({"input": test_query})
    assert "output" in response
    assert len(response["output"]) > 0


if __name__ == "__main__":
    # Run with verbose output
    print("🧪 Running DeepEval tests...")
    print("=" * 50)

    # When ready, run with:
    # deepeval test run exercises/exercise_2_eval.py -v

    # For now, run basic pytest:
    pytest.main([__file__, "-v"])
