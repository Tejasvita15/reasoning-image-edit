# src/taxonomy.py

CATEGORY_TO_DIMENSION = {

    # Factual knowledge
    "anomaly_correction": "attribute_perception",
    "color_change": "attribute_perception",
    "count_change": "attribute_perception",
    "part_completion": "attribute_perception",
    "size_adjustment": "attribute_perception",

    "position_movement": "spatial_perception",
    "viewpoint_change": "spatial_perception",

    "temporal_prediction": "temporal_prediction",

    # Conceptual knowledge
    "humanities": "social_science",
    "practical_knowledge": "social_science",

    "biology": "natural_science",
    "chemistry": "natural_science",
    "geography": "natural_science",
    "mathematics": "natural_science",
    "medicine": "natural_science",
    "physics": "natural_science",

    # Procedural knowledge
    "abstract_reasoning": "logical_reasoning",
    "rule-based_reasoning": "logical_reasoning",

    "multi-element_composition": "instruction_decomposition",
    "multi-instruction_execution": "instruction_decomposition",
}


DIMENSION_TO_KNOWLEDGE = {
    "attribute_perception": "factual",
    "spatial_perception": "factual",
    "temporal_prediction": "factual",

    "social_science": "conceptual",
    "natural_science": "conceptual",

    "logical_reasoning": "procedural",
    "instruction_decomposition": "procedural",
}


# Categories that are not eligible for the RQ1 Step1X comparison.
#
# Step1X is being evaluated only on single-reference image editing.
# multi-element_composition contains multi-reference examples and
# should therefore be excluded from the eligible pool.
RQ1_EXCLUDED_CATEGORIES = {
    "multi-element_composition",
}


def get_reasoning_dimension(category: str) -> str:
    """
    Map a KRIS-Bench native category to one of the
    seven reasoning dimensions.
    """

    if category not in CATEGORY_TO_DIMENSION:
        raise ValueError(
            f"Unknown KRIS-Bench category: {category}"
        )

    return CATEGORY_TO_DIMENSION[category]


def get_knowledge_type(category: str) -> str:
    """
    Map a KRIS-Bench native category to the primary
    RQ1 knowledge type: factual, conceptual, or procedural.
    """

    dimension = get_reasoning_dimension(category)

    return DIMENSION_TO_KNOWLEDGE[dimension]


def is_rq1_category_allowed(category: str) -> bool:
    """
    Return whether the category is eligible for the RQ1
    single-reference experiment.

    Note:
    We will still verify the actual number of source images
    per example in dataset.py. This category-level check is
    only an additional safeguard.
    """

    if category not in CATEGORY_TO_DIMENSION:
        raise ValueError(
            f"Unknown KRIS-Bench category: {category}"
        )

    return category not in RQ1_EXCLUDED_CATEGORIES