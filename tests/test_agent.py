"""
Test suite for the Oak Brook IT Support Agent using DeepEval.

These tests use LLM-as-a-judge to evaluate semantic quality rather than exact matches.
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from src.agent import build_oak_brook_support_agent


# Initialize metrics with specific passing thresholds
answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.8)


# Knowledge base context for faithfulness evaluation
KNOWLEDGE_BASE_CONTEXT = [
    "Healthcare IT Support: We ensure all HIPAA regulations regarding technology are followed. Our proactive remote and onsite support locks down patient data.",
    "Cybersecurity Services: We provide advanced solutions including firewalls, threat detection, employee training, and continuous vulnerability assessments.",
    "Cloud Services: We offer migration, management, and optimization of cloud infrastructure across AWS, Azure, and Google Cloud platforms.",
    "Data Backup & Recovery: Automated daily backups with 99.9% recovery success rate. RTO under 4 hours for critical systems.",
    "Network Infrastructure: Design, implementation, and 24/7 monitoring of enterprise networks with 99.99% uptime SLA."
]


@pytest.mark.parametrize(
    "user_query",
    [
        "What healthcare IT services do you offer?",
        "Tell me about your cybersecurity solutions",
        "Do you provide cloud migration services?",
        "What backup options are available?",
    ]
)
def test_oak_brook_support_agent_quality(user_query):
    """
    Test the agent's output quality using DeepEval metrics.
    
    This test evaluates:
    - Answer Relevancy: Does the answer address the user's question?
    - Faithfulness: Is the answer grounded in the knowledge base?
    """
    # Build and run the agent
    executor = build_oak_brook_support_agent()
    response = executor.invoke({"input": user_query})
    actual_output = response["output"]
    
    # Construct the DeepEval Test Case
    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_output,
        retrieval_context=KNOWLEDGE_BASE_CONTEXT
    )
    
    # Execute the automated assertions
    assert_test(test_case, [answer_relevancy, faithfulness])


@pytest.mark.skip(reason="Run manually to test warranty tool integration")
def test_warranty_check_integration():
    """Test that warranty checking works correctly."""
    executor = build_oak_brook_support_agent()
    
    # Test VIP customer
    response = executor.invoke({
        "input": "Customer OAK-VIP-1234 needs help with hardware issues"
    })
    assert "premium" in response["output"].lower() or "SLA" in response["output"]
    
    # Test standard customer
    response = executor.invoke({
        "input": "Customer OAK-STD-5678 is having software problems"
    })
    assert "standard" in response["output"].lower()