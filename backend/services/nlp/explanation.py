TONE_LABELS = {
    "neutral": "neutral",
    "nostalgia": "nostalgic",
    "horror": "horror",
    "romantic": "romantic",
    "academic": "academic",
    "joyful": "joyful",
    "melancholic": "melancholic",
    "hopeful": "hopeful",
    "mysterious": "mysterious",
    "formal": "formal",
}


def generate_explanation(
    context: str,
    description: str,
    words: list[str],
    mode: str = "write",
    blank_present: bool = False,
    selection_present: bool = False,
    intent: str = "sentence",
) -> str:
    if not words:
        return "Unable to generate suggestions at the moment."

    tone = TONE_LABELS.get(context.lower(), context) if context else "the selected"

    if selection_present:
        return f"Selection-focused suggestions ranked by semantic fit and {tone} tone."

    if blank_present:
        return (
            "Blank-fill suggestions are grammar-filtered for the missing slot "
            f"and aligned to a {tone} tone."
        )

    if mode == "edit":
        return "Polish mode prioritizes grammar safety and clarity with controlled tone."

    if mode == "rewrite":
        return (
            f"Transform mode keeps sentence meaning while shifting style toward {tone} tone."
        )

    if intent == "sentence":
        return f"Draft suggestions ranked by grammar fit, semantic match, and {tone} tone."

    return f"Suggestions matched to your sentence intent in a {tone} tone."
