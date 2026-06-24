"""You can define your own user defined functions here to apply dataset transformations.
A UDF function must take a row of the dataset as its first positional argument, plus optional kwargs, and return the transformed row.

This can be used for lightweight dataset transformations. A common use case is if your dataset contains documents and you want to format them into a prompt template.
"""
from typing import Dict, Any


def format_propella_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
    """Example UDF; requires optional inference_hive.propella (not shipped in this repo)."""
    from inference_hive.propella import create_messages  # noqa: PLC0415

    row['messages'] = create_messages(row['text'])
    return row