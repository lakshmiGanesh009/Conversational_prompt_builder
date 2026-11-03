"""Static parameter catalog for the Conversational Prompt Builder.

This module holds the small, static configuration describing which parameters
exist, which are required, and their default values. It is pure data plus a few
read-only helpers. The Intent_Analyzer and Elicitation_Engine both read this
catalog to decide which parameters to extract, which are still missing, and what
default values to offer as suggestions (design: "Model notes", Requirements 2.4
and 3.5).

Keeping the catalog as data rather than code makes it easy to adjust the set of
parameters without touching the deterministic logic.
"""

from __future__ import annotations

from typing import Optional

from prompt_builder.models import ParameterDefinition

# The static set of parameters the app understands. These are goal-supporting
# attributes of the prompt (design Data Models: "audience, tone, format, length,
# domain, ..."). Some are required; others carry a sensible default value that
# the Elicitation_Engine offers as a suggestion (Req 3.5).
CATALOG: list[ParameterDefinition] = [
    # Who the output is for. Required: strongly shapes the prompt and has no
    # single sensible default.
    ParameterDefinition(name="audience", required=True),
    # The subject area / domain the prompt operates in. Required for the same
    # reason as audience.
    ParameterDefinition(name="domain", required=True),
    # The voice of the output. Optional with a neutral default.
    ParameterDefinition(name="tone", required=False, default_value="neutral"),
    # The shape of the output. Optional, defaulting to flowing prose.
    ParameterDefinition(name="format", required=False, default_value="prose"),
    # How long the output should be. Optional, defaulting to a medium length.
    ParameterDefinition(name="length", required=False, default_value="medium"),
]

# Index by name for O(1) lookups. Built once at import time from CATALOG.
_BY_NAME: dict[str, ParameterDefinition] = {
    definition.name: definition for definition in CATALOG
}


def required_parameter_names() -> list[str]:
    """Return the names of the required parameters, in catalog order.

    The Intent_Analyzer uses this (together with the collected values) to derive
    the set of still-missing required parameters (Req 2.4).
    """
    return [definition.name for definition in CATALOG if definition.required]


def get_definition(name: str) -> Optional[ParameterDefinition]:
    """Return the ParameterDefinition for ``name``, or ``None`` if undefined."""
    return _BY_NAME.get(name)
