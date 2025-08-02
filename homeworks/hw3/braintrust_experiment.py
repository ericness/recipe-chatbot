#!/usr/bin/env python3
"""
Braintrust Experiment Script for Recipe Chatbot

This script:
1. Uses the "dietary-queries" dataset from Braintrust
2. Runs an experiment that calls get_agent_response for each query
3. Uses temperature=0.5 to introduce variation in responses
4. Logs all results to Braintrust for analysis using the Eval framework
"""

import os
import sys
from pathlib import Path

from braintrust import Eval, init_dataset
from dotenv import load_dotenv

# Add the project root to the Python path so we can import from backend
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.utils import get_agent_response

# Load environment variables
load_dotenv()


def recipe_task(input_data):
    """Task function that calls get_agent_response."""
    # Extract the query text from the input structure
    if isinstance(input_data, dict) and "query" in input_data:
        query_text = input_data["query"]
    else:
        query_text = str(input_data)

    messages = [{"role": "user", "content": query_text}]

    response_messages = get_agent_response(
        messages=messages, metadata={"temperature": 0.5}, temperature=0.5
    )

    return response_messages[-1]["content"]


# Run the evaluation using the Eval framework
Eval(
    "recipe-chatbot",
    data=init_dataset(
        project=os.environ.get("BRAINTRUST_PROJECT", "recipe-chatbot"),
        name="dietary-queries",
    ),
    task=recipe_task,
    scores=[],  # No automatic scoring for now, just collecting responses
    trial_count=3,  # Run each dataset member 3 times to measure variance
)
