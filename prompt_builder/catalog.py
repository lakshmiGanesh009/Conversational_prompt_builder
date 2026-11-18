"""Static, backward-compatible prompt brief parameter catalog.

Only ``audience`` and ``domain`` retain their original global-required status.
The richer fields are optional globally and become question candidates only when
the adaptive planner determines they materially help the current task.
"""

from __future__ import annotations

from typing import Optional

from prompt_builder.models import ParameterDefinition


def _p(
    name: str,
    *,
    required: bool = False,
    default: Optional[str] = None,
    label: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "Brief",
) -> ParameterDefinition:
    return ParameterDefinition(name, required, default, label, description, category)


CATALOG: list[ParameterDefinition] = [
    # Original catalog entries stay first and preserve required/default behavior.
    _p("audience", required=True, label="Audience", description="Who will use or read the result"),
    _p("domain", required=True, label="Domain", description="Subject area and operating context"),
    _p("tone", default="neutral and professional", label="Tone", description="Emotional register and attitude"),
    _p("format", default="clear structured text", label="Output format", description="Deliverable shape or schema"),
    _p("length", default="appropriate to the task", label="Length / depth", description="Expected response detail"),
    # General prompt brief.
    _p("objective", label="Desired outcome", description="The concrete result the response must achieve"),
    _p("background_context", label="Background context", description="Facts and situation the model needs"),
    _p("role", label="Model role", description="Expertise or perspective the model should assume"),
    _p("input_material", label="Inputs / source material", description="Content, data, or facts supplied at run time"),
    _p("deliverable", label="Deliverable", description="The artifact the model must produce"),
    _p("output_schema", label="Output schema", description="Exact fields, sections, or machine-readable contract"),
    _p("voice", label="Voice", description="Writing personality and phrasing conventions"),
    _p("language", default="English", label="Language", description="Response language"),
    _p("reading_level", label="Reading level", description="Vocabulary and complexity target"),
    _p("platform", label="Platform / application", description="Where the result will be used"),
    _p("must_include", label="Must include", description="Mandatory content or behaviors"),
    _p("exclusions", label="Exclusions", description="Content or behavior to avoid"),
    _p("constraints", label="Constraints", description="Hard limits, policies, and boundaries"),
    _p("security_compliance", label="Safety / compliance", description="Security, legal, policy, or privacy requirements"),
    _p("examples", label="Examples / references", description="Style samples or target examples"),
    _p("success_criteria", label="Definition of done", description="Measurable acceptance criteria"),
    _p("factuality", default="Do not invent facts; identify uncertainty", label="Factuality", description="Accuracy and evidence expectations"),
    _p("citation_expectations", label="Sources / citations", description="Citation format and source quality expectations"),
    _p("edge_cases", label="Edge cases", description="Failure behavior, ambiguity, and exceptions"),
    _p("assumptions", label="Assumptions", description="Defaults the model may safely make"),
    _p("self_check", label="Quality checks", description="Checks to run before returning the answer"),
    _p("model_settings", label="Model settings", description="Useful inference guidance"),
    # Software / product.
    _p("tech_stack", label="Stack / preferences", description="Languages, frameworks, storage, and tooling", category="Software"),
    _p("feature_scope", label="Feature scope", description="In-scope and out-of-scope capabilities", category="Software"),
    _p("features", label="Features", description="Requested product capabilities", category="Software"),
    _p("user_roles", label="Roles / permissions", description="Actors and authorization boundaries", category="Software"),
    _p("auth", label="Authentication", description="Identity and session requirements", category="Software"),
    _p("data_integrations", label="Data / integrations", description="Entities, APIs, imports, and external systems", category="Software"),
    _p("nonfunctional", label="Non-functional needs", description="Performance, reliability, testing, and operations", category="Software"),
    _p("accessibility", default="WCAG 2.2 AA where applicable", label="Accessibility", description="Inclusive interaction requirements", category="Software"),
    _p("deployment", label="Deployment", description="Target environment, hosting, and release needs", category="Software"),
    _p("structure", label="Structure", description="Required organization of the deliverable"),
    # Marketing / commerce.
    _p("product", label="Product / service", description="What is being promoted", category="Marketing"),
    _p("offer", label="Offer", description="Value proposition, price, incentive, or deal", category="Marketing"),
    _p("channel", label="Channel", description="Email, web, paid social, print, or another placement", category="Marketing"),
    _p("call_to_action", label="Call to action", description="The action the audience should take", category="Marketing"),
    _p("proof_points", label="Proof", description="Evidence, differentiators, testimonials, or approved claims", category="Marketing"),
    # Learning / notes.
    _p("learner_level", label="Learner level", description="Existing knowledge and expertise", category="Learning"),
    _p("duration", label="Duration", description="Available time or content duration", category="Learning"),
    _p("assessment", label="Assessment", description="Practice, quiz, rubric, or mastery check", category="Learning"),
    # Video.
    _p("aspect_ratio", default="9:16 for short-form video", label="Aspect ratio", description="Canvas dimensions and orientation", category="Video"),
    _p("pacing_style", label="Pacing / style", description="Rhythm, edit style, and visual treatment", category="Video"),
    _p("captions_audio", label="Captions / audio", description="Subtitle, voice-over, music, and sound requirements", category="Video"),
    _p("export_specs", label="Export specs", description="Codec, dimensions, frame rate, and delivery constraints", category="Video"),
    _p("style", label="Style", description="Visual or editorial style"),
]

_BY_NAME: dict[str, ParameterDefinition] = {definition.name: definition for definition in CATALOG}


def required_parameter_names() -> list[str]:
    """Return globally required names in stable catalog order."""
    return [definition.name for definition in CATALOG if definition.required]


def get_definition(name: str) -> Optional[ParameterDefinition]:
    """Return the definition for ``name``, or ``None`` for an unknown field."""
    return _BY_NAME.get(name)


def parameter_label(name: str) -> str:
    """Return a human-friendly label for a catalog field."""
    definition = get_definition(name)
    return (definition.label if definition and definition.label else name.replace("_", " ").title())
