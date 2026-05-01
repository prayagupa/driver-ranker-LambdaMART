# PRD: Rider-Driver Matching with ML Ranking & Real-Time Features

## 1. Overview

### Problem Statement
When a rider requests a ride, the system must select the optimal driver from a pool of available candidates. Today's rule-based assignment (nearest driver wins) leaves significant value on the table—ignoring driver quality, rider preferences, ETA accuracy, and platform-level objectives like utilization and cancellation reduction.

### Objective
Replace the heuristic assignment logic with a **machine-learning ranking model** that scores every eligible driver for a given ride request and selects the best match, powered by **real-time features** served at low latency.

### Success Metrics
| Metric | Baseline | Target |
|--------|----------|--------|
| Rider cancellation rate | 8.2% | ≤ 5.5% |
| Driver acceptance rate | 74% | ≥ 85% |
| Average rider wait time | 4.8 min | ≤ 4.2 min |
| Completed rides / driver-hour | 2.1 | ≥ 2.4 |
| Rider NPS (post-trip) | 62 | ≥ 68 |

---

## 2. Scope

### In Scope
- ML ranking model design, training, and serving
- Real-time feature pipeline (ingestion → computation → serving)
- Integration with the dispatch/matching service
- A/B experimentation framework hooks
- Monitoring & observability

### Out of Scope
- Pricing/surge logic (consumed as a feature, not owned)
- Driver onboarding & compliance
- Rider-facing UX changes beyond ETA display

---

## 3. System Architecture

```
┌────────────┐      ┌──────────────────┐      ┌────────────────┐
│ Ride Request│─────▶│  Matching Service │─────▶│  ML Rank Model │
└────────────┘      └──────────────────┘      └────────────────┘
                           │      ▲                     ▲
                           ▼      │                     │
                    ┌─────────────────┐        ┌───────────────┐
                    │ Candidate Filter │        │ Feature Store  │
                    │ (geo radius, status)     │ (real-time +   │
                    └─────────────────┘        │  batch)        │
                                               └───────────────┘
```

---

## 4. Real-Time Feature Pipeline

### 4.1 Feature Categories

| Category | Examples | Freshness | Source |
|----------|----------|-----------|--------|
| **Driver real-time** | Current location, speed, heading, idle time, active trip status | < 2 s | GPS stream, trip events |
| **Driver historical** | Acceptance rate (7d), avg rating, cancellation rate, trips completed | Hourly batch | Data warehouse |
| **Rider real-time** | Pickup location, destination, surge multiplier, payment method | Per-request | Ride request payload |
| **Rider historical** | Avg tip %, rating given, cancellation history | Hourly batch | Data warehouse |
| **Contextual** | Time of day, day of week, weather, local event flags, traffic index | 1–5 min | External APIs, streaming |
| **Pair/interaction** | Historical trips between this rider-driver pair, route familiarity | Hourly batch | Trip logs |

### 4.2 Feature Computation & Serving

- **Stream processing**: Apache Kafka / Flink for GPS, trip-state, and event streams → sliding window aggregates (e.g., driver idle time last 5 min).
- **Online store**: Low-latency key-value store (Redis / DynamoDB) serving pre-computed features at p99 < 5 ms.
- **Batch store**: Spark jobs materialize historical features into the online store on an hourly/daily cadence.
- **Feature registry**: Central catalog (feature name, owner, SLA, lineage) enabling discoverability and governance.

---

## 5. ML Ranking Model

### 5.1 Model Formulation
- **Task**: Learning-to-rank (LTR) — given a rider request and N candidate drivers, produce a relevance score for each candidate.
- **Approach**: Pairwise/listwise LambdaMART (gradient-boosted trees) for v1; transformer-based cross-encoder for v2.
- **Label**: Composite outcome label derived from:
  - Driver accepted (binary)
  - Trip completed without cancellation (binary)
  - Rider rating ≥ 4.5 (binary)
  - Weighted combination → continuous relevance score

### 5.2 Training Pipeline
1. Extract historical ride requests with candidate pools (offline replay of dispatch logs).
2. Join features point-in-time correctly to avoid leakage.
3. Train with cross-validation; optimize NDCG@5.
4. Register model artifact in model registry with lineage metadata.

### 5.3 Serving
- Model exported to ONNX / TensorRT for inference.
- Served via a sidecar or dedicated inference service (< 20 ms p99 for scoring 50 candidates).
- Fallback: If model latency exceeds SLA, fall back to lightweight heuristic scorer.

---

## 6. Matching Flow (End-to-End)

1. **Ride request received** — extract rider features from request + feature store.
2. **Candidate generation** — geo-filter drivers within dynamic radius (expands if pool too small).
3. **Feature hydration** — batch-fetch driver features for all candidates from online store.
4. **Model scoring** — rank candidates; return ordered list with scores.
5. **Business rule overlay** — apply hard constraints (e.g., vehicle type match, accessibility needs, fraud flags).
6. **Dispatch** — offer trip to top-ranked driver; on decline/timeout, cascade to next.

---

## 7. Experimentation & Rollout

| Phase | Population | Duration | Gate Criteria |
|-------|-----------|----------|---------------|
| Shadow mode | 100% (log predictions, no action) | 2 weeks | Offline NDCG ≥ 0.78 |
| Canary | 5% of requests | 1 week | No regression in wait time or cancellation |
| Ramp | 25% → 50% → 100% | 3 weeks | Target metrics trending positive |

- Experiment assignment at the **request level** (rider-session hash).
- Guardrail metrics: p99 latency, error rate, earnings fairness across driver cohorts.

---

## 8. Monitoring & Observability

| Signal | Tool | Alert Threshold |
|--------|------|-----------------|
| Feature freshness (staleness) | Feature store health dashboard | > 30 s for real-time features |
| Model latency p99 | APM (Datadog / Prometheus) | > 25 ms |
| Prediction distribution drift | Evidently / custom | KL divergence > 0.1 over 1 h |
| Online metric degradation | Experimentation platform | > 1% regression in acceptance rate |
| Feature store availability | Uptime monitor | < 99.95% over rolling 5 min |

---

## 9. Fairness & Compliance

- **Driver earnings equity**: Monitor earnings distribution across demographic proxies; alert on Gini coefficient increase > 5%.
- **Bias audit**: Quarterly offline analysis of model scores segmented by driver tenure, region, vehicle age.
- **Explainability**: SHAP values logged per prediction for audit trail; top-3 contributing features surfaced in internal tooling.

---

## 10. Dependencies & Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Feature store latency spike | Medium | High (dispatch delay) | Circuit breaker + heuristic fallback |
| Cold-start for new drivers | High | Medium (unfair low ranking) | Exploration bonus in score; separate onboarding pool |
| Label noise (rider ratings unreliable) | Medium | Medium | Multi-signal label; down-weight noisy labels |
| Model staleness after distribution shift | Medium | High | Weekly retraining; drift detection triggers ad-hoc retrain |

---

## 11. Milestones & Timeline

| Milestone | Target Date |
|-----------|-------------|
| Feature pipeline v1 (real-time + batch) in staging | Week 4 |
| Model v1 trained & validated offline | Week 6 |
| Shadow-mode deployment | Week 7 |
| Canary experiment launch | Week 9 |
| Full production rollout | Week 12 |
| Model v2 (transformer ranker) exploration | Week 16 |

---

## 12. Team & Ownership

| Role | Owner |
|------|-------|
| Product Manager | TBD |
| ML Engineer (model) | TBD |
| Data Engineer (feature pipeline) | TBD |
| Backend Engineer (matching service) | TBD |
| Data Scientist (experimentation) | TBD |

---

## Appendix: Candidate Feature List (v1)

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | ETA to pickup (seconds) | Real-time | Routing engine |
| 2 | Driver idle duration (seconds) | Real-time | GPS stream |
| 3 | Driver acceptance rate (7d) | Batch | Trip logs |
| 4 | Driver avg rating (30d) | Batch | Ratings table |
| 5 | Driver cancellation rate (7d) | Batch | Trip logs |
| 6 | Driver trips completed (lifetime) | Batch | Trip logs |
| 7 | Rider avg rating given | Batch | Ratings table |
| 8 | Rider cancellation rate (30d) | Batch | Trip logs |
| 9 | Rider avg tip % | Batch | Payments |
| 10 | Surge multiplier at pickup | Real-time | Pricing service |
| 11 | Time of day (cyclical encoding) | Contextual | Clock |
| 12 | Day of week (one-hot) | Contextual | Clock |
| 13 | Weather condition (categorical) | Contextual | Weather API |
| 14 | Historical pair trip count | Batch | Trip logs |
| 15 | Driver route familiarity score | Batch | GPS history |
