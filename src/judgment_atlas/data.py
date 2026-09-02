from __future__ import annotations

import numpy as np
import pandas as pd

from . import SEED

CONTEXTS = {
    "routine": ("find a routine service option", "standard service details and operating hours"),
    "definition": ("define the requested technical term", "descriptive definition with applied examples"),
    "ambiguous": ("interpret a multi-meaning information need", "contextual passage using an alternate sense"),
    "policy": ("apply a strict priority-tier policy", "candidate near a protected tier boundary"),
}
MODELS = ["cautious_judge", "lenient_judge", "policy_brittle_judge"]
PERSONAS = ["neutral", "efficiency_focused", "accessibility_focused"]


def generate_judgments(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Generate fictional judge logs; no outputs come from a real LLM or person."""
    rng = np.random.default_rng(seed)
    rows = []
    contexts = list(CONTEXTS)
    for i in range(n):
        context = rng.choice(contexts, p=[.32, .23, .20, .25])
        model = rng.choice(MODELS)
        persona = rng.choice(PERSONAS)
        intervention = rng.choice(["label", "explanation", "independent_first"], p=[.34, .33, .33])
        lexical_overlap = float(rng.beta(2.4, 2.2))
        ambiguity = float(np.clip(rng.normal({"routine":.15,"definition":.42,"ambiguous":.82,"policy":.58}[context], .13), 0, 1))
        boundary_distance = float(rng.beta(1.5, 4.5) if context == "policy" else rng.beta(4, 2))
        human_label = int(rng.random() < (.30 + .42 * lexical_overlap - .10 * ambiguity))
        error_logit = -3.0 + 2.1 * ambiguity + 1.35 * (1-boundary_distance)
        error_logit += .80 * (context == "definition") + 1.05 * (context == "policy")
        error_logit += .70 * (model == "policy_brittle_judge") + .35 * (model == "lenient_judge")
        error_logit += .65 * (persona == "accessibility_focused") * (context == "policy")
        error_logit -= .55 * (intervention == "independent_first")
        error_probability = 1 / (1 + np.exp(-error_logit))
        disagreement = int(rng.random() < error_probability)
        judge_label = human_label ^ disagreement
        q, d = CONTEXTS[context]
        query = f"{q}; case {i % 37}"
        document = f"{d}; overlap {int(lexical_overlap*10)}; frame {i % 19}"
        rows.append((f"sim-{i:06d}", i//250, context, model, persona, intervention, query, document, lexical_overlap, ambiguity, boundary_distance, human_label, judge_label, disagreement))
    return pd.DataFrame(rows, columns=["event_id","batch","context","model","persona","intervention","query","document","lexical_overlap","ambiguity","boundary_distance","reference_label","judge_label","disagreement"])
