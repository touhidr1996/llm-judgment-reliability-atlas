# Model card

## Intended use

The classifier predicts which simulated judge events are likely to disagree with a simulated reference so a limited verification budget can be allocated. It demonstrates selective verification, not automated truth determination.

## Model and evaluation

A seeded histogram gradient-boosting classifier uses context, fictional judge/persona/intervention labels, lexical overlap, ambiguity, and policy-boundary distance. The final four batches are held out chronologically. A constant global-risk baseline is compared with the cluster-aware feature model using PR-AUC, Brier score, ROC-AUC, error capture at 25% review, and residual error.

## Limitations and governance

The synthetic mechanism makes evaluation optimistic and cannot validate performance on real LLMs. Reference labels are assumed available and correct. A real study would require preregistered prompts, repeated model trials, human annotation protocols, uncertainty intervals, subgroup and intersectional analyses, privacy review, model-version logging, appeal processes, and an independent policy verifier.
