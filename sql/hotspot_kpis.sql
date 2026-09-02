-- Operational audit: review load and residual disagreement by simulated batch.
SELECT batch,
       COUNT(*) AS decisions,
       ROUND(AVG(review_required),4) AS review_rate,
       ROUND(AVG(CASE WHEN review_required=0 THEN disagreement END),4) AS residual_disagreement,
       ROUND(AVG(disagreement_risk),4) AS mean_predicted_risk
FROM fact_triage
GROUP BY batch
ORDER BY batch;
