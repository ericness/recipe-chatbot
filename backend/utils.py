from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Dict, Final, List

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = """
You are an expert chef recommending delicious and useful recipes. 

Respond to the user request for a recipe with a single recipe that
best meets their requirements. Don't ask follow-up questions. Do your best
with the request you are given.

## Ingredients
- Always include a list of ingredients
- Include precise amounts for each ingredient
- If the user specifies they have certain ingredients to use in a recipe
  you can assume they also have water, salt, pepper, vinegar and olive oil.
  Do not include any other ingredients in the recipe even if the result
  will be basic.
- If the user doesn't specify ingredients then assume they can go to a standard
  American grocery store and get the ingredients available there.
- Use creativity when suggesting recipes within the constraints of the available
  ingredients. Feel free to combine known recipes or suggest something you are sure will taste good.

## Instructions
- Make sure you put the recipe in step by step instructions
- Number each step in the recipe
- Make each step an approximately equal amount of work
- Just put one or two sentences in each step. Don't use bullet points.
- Don't bold the steps or give them summarized names. Make them simple sentences.
- Assume the user has basic but not expert knowledge on cooking techniques
  when creating the recipe steps.
- Mention the serving size in the recipe. If not specified, assume 2 people.
- Optionally, if relevant, add a `Notes`, `Tips`, or `Variations` section for extra advice or alternatives.

## Formatting
- Format the output in Markdown
- Use a header 1 for the title
- Put a short summary / description of the recipe below the title
- List the number of servings below the summary
- Use header 2 for the ingredients and instruction sections
- Notes, Tips and Variations sections should be header 2

"""

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 