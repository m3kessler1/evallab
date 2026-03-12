"""
Tool definitions for the Oak Brook IT Support Agent.

These tools demonstrate how to use Pydantic schemas to enforce strict input validation,
acting as deterministic guardrails for the non-deterministic LLM.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal


class WarrantyCheckInput(BaseModel):
    """Schema for warranty check tool inputs."""
    customer_id: str = Field(
        ..., 
        description="The unique identifier for the customer, typically starting with 'OAK-'."
    )
    issue_category: Literal["hardware", "software", "network"] = Field(
        ..., 
        description="The technical category of the reported issue."
    )


@tool("check_warranty_status", args_schema=WarrantyCheckInput)
def check_warranty_status(customer_id: str, issue_category: str) -> str:
    """
    Queries the backend billing system to check if the customer has an active SLA or warranty contract.
    
    Args:
        customer_id: The customer's unique identifier
        issue_category: The category of the technical issue
    
    Returns:
        String indicating warranty status and next steps
    """
    if customer_id.startswith("OAK-VIP"):
        return f"Customer {customer_id} holds an active premium Service Level Agreement for {issue_category} support. Proceed with full technical assistance."
    elif customer_id.startswith("OAK-STD"):
        return f"Customer {customer_id} has a standard contract. Ensure the {issue_category} issue is covered under basic support before proceeding."
    return f"Customer ID {customer_id} not found or out of warranty. Escalate to the sales team immediately and do not provide technical solutions."


@tool("search_oak_brook_kb")
def search_oak_brook_kb(query: str) -> str:
    """
    Searches the Oak Brook IT Solutions proprietary knowledge base for service information and protocols.
    
    Args:
        query: The search query string
    
    Returns:
        Relevant knowledge base article or not found message
    """
    kb_data = {
        "healthcare": "Healthcare IT Support: We ensure all HIPAA regulations regarding technology are followed. Our proactive remote and onsite support locks down patient data.",
        "cybersecurity": "Cybersecurity Services: We provide advanced solutions including firewalls, threat detection, employee training, and continuous vulnerability assessments.",
        "cloud": "Cloud Services: We offer migration, management, and optimization of cloud infrastructure across AWS, Azure, and Google Cloud platforms.",
        "backup": "Data Backup & Recovery: Automated daily backups with 99.9% recovery success rate. RTO under 4 hours for critical systems.",
        "network": "Network Infrastructure: Design, implementation, and 24/7 monitoring of enterprise networks with 99.99% uptime SLA."
    }
    
    query_lower = query.lower()
    for key, info in kb_data.items():
        if key in query_lower:
            return info
    
    return "No specific information found in the knowledge base regarding this query. Please consult with a senior technician."