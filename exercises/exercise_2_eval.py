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
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

# TODO: Import the metrics you want to use
# from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from src.agent import build_oak_brook_support_agent


# TODO: Initialize metrics with specific thresholds
# answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
# faithfulness = FaithfulnessMetric(threshold=1.0)


# Knowledge base context for faithfulness evaluation
KNOWLEDGE_BASE_CONTEXT = [
    "Healthcare IT Support: We ensure all HIPAA regulations regarding technology are followed.",
    "Cybersecurity Services: We provide advanced solutions including firewalls, threat detection, and vulnerability assessments.",
    "Cloud Services: We offer migration, management, and optimization of cloud infrastructure.",
    "Data Backup & Recovery: Automated daily backups with 99.9% recovery success rate.",
    "Network Infrastructure: Design, implementation, and 24/7 monitoring with 99.99% uptime SLA."
]


# TODO: Implement the test function
# @pytest.mark.parametrize(
#     "user_query,expected_topics",
#     [
#         ("What healthcare IT services do you offer?", ["HIPAA", "healthcare", "patient data"]),
#         ("Tell me about cybersecurity", ["firewalls", "threat detection", "vulnerability"]),
#         ("Do you do cloud migration?", ["cloud", "migration", "AWS", "Azure"]),
#     ]
# )
# def test_oak_brook_support_agent_quality(user_query, expected_topics):
#     """
#     Test the agent's output quality using DeepEval metrics.
#     
#     This test evaluates:
#     - Answer Relevancy: Does the answer address the user's question?
#     - Faithfulness: Is the answer grounded in the knowledge base?
#     """
#     # Build and run the agent
#     executor = build_oak_brook_support_agent()
#     response = executor.invoke({"input": user_query})
#     actual_output = response["output"]
#     
#     # Create the test case
#     test_case = LLMTestCase(
#         input=user_query,
#         actual_output=actual_output,
#         retrieval_context=KNOWLEDGE_BASE_CONTEXT
#     )
#     
#     # Assert against the metrics
#     assert_test(test_case, [answer_relevancy, faithfulness])


# Starter test without DeepEval (remove this when you implement above)
def test_agent_runs():
    """Basic smoke test to ensure agent executes without errors."""
    executor = build_oak_brook_support_agent()
    response = executor.invoke({"input": "What services do you offer?"})
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