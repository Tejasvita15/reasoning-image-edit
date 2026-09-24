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


def get_reasoning_dimension(category):
    if category not in CATEGORY_TO_DIMENSION:
        raise ValueError(
            f"Unknown KRIS-Bench category: {category}"
        )

    return CATEGORY_TO_DIMENSION[category]


def get_knowledge_type(category):
    dimension = get_reasoning_dimension(category)
    return DIMENSION_TO_KNOWLEDGE[dimension]