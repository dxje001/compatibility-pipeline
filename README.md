# Compatibility Scoring System: Final Report

**Project**: Big Data Compatibility Pipeline
**Date**: January 2026
**Status**: Complete (Iterations 1-3)

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Data Sources](#2-data-sources)
3. [Modeling Approach](#3-modeling-approach)
4. [Late Fusion Architecture](#4-late-fusion-architecture)
5. [Stability Analysis](#5-stability-analysis)
6. [UI Questionnaire Design](#6-ui-questionnaire-design)
7. [Inference & UI Implementation](#7-inference--ui-implementation)
8. [Limitations](#8-limitations)
9. [Future Work](#9-future-work)
10. [Appendix: Artifacts](#appendix-artifacts)

---

## 1. Problem Definition

### 1.1 Compatibility as a Latent Variable

Compatibility between two people is not directly observable. Unlike classification tasks with ground-truth labels, compatibility is a **latent variable** that:

- Cannot be measured objectively
- Varies by context (romantic, professional, friendship)
- Is influenced by factors that may not be captured in survey data
- Has no universally agreed-upon definition

This fundamental characteristic shapes every design decision in this project. We do not claim to predict "true" compatibility; instead, we build a scoring system that:

1. Is **internally consistent** (stable across random seeds)
2. Is **theoretically motivated** (based on personality psychology research)
3. Is **interpretable** (users understand what drives the score)
4. Is **reproducible** (fully deterministic given configuration)

### 1.2 Motivation and Real-World Relevance

Compatibility scoring has applications in:

- **Dating platforms**: Matching users based on personality and interests
- **Team formation**: Assembling complementary work teams
- **Roommate matching**: University housing assignments
- **Mentorship programs**: Pairing mentors with mentees

The core challenge is identical across domains: given profiles of two people, produce a meaningful compatibility score without access to outcome data (e.g., "did they become a successful couple?").

### 1.3 Project Scope

This project delivers:

| Component | Description |
|-----------|-------------|
| Offline training pipeline | Trains two independent compatibility models |
| Stability analysis | Validates model consistency across random seeds |
| UI questionnaire | 10-item survey derived from feature importance |
| Inference engine | Transforms survey responses into scores |
| Web interface | Streamlit application for end-user interaction |

---

## 2. Data Sources

### 2.1 IPIP Big Five Dataset

**Source**: International Personality Item Pool (IPIP)
**Size**: ~1 million respondents
**Format**: 50 Likert-scale questions (1-5)

The IPIP-50 measures the Big Five personality dimensions (OCEAN):

| Dimension | Items | Description |
|-----------|-------|-------------|
| Extraversion | EXT1-EXT10 | Sociability, assertiveness, positive emotions |
| Neuroticism | EST1-EST10 | Anxiety, emotional instability, negative affect |
| Agreeableness | AGR1-AGR10 | Cooperation, trust, altruism |
| Conscientiousness | CSN1-CSN10 | Organization, dependability, self-discipline |
| Openness | OPN1-OPN10 | Creativity, curiosity, openness to experience |

**Preprocessing**:
- Reverse scoring applied to designated items
- Missing values imputed with column means
- OCEAN aggregates computed as item means
- Features standardized (z-score normalization)

### 2.2 OkCupid Profiles Dataset

**Source**: OkCupid (publicly released research dataset)
**Size**: ~60,000 profiles
**Format**: Mixed (categorical, numeric, text)

| Feature Type | Columns | Description |
|--------------|---------|-------------|
| Numeric | age, height, income | Continuous attributes |
| Categorical | religion, education, drinks, smokes, drugs, status, offspring, etc. | Discrete attributes |
| Text | essay0-essay9 | Free-text self-descriptions |

**Preprocessing**:
- Numeric: Median imputation, z-score normalization
- Categorical: Missing → "unknown", one-hot encoding
- Text: Concatenation, TF-IDF vectorization (max 5,000 features, 1-2 ngrams)

### 2.3 Why No Direct Labels Exist

Neither dataset contains compatibility outcomes. We cannot observe:

- Whether two IPIP respondents would be compatible
- Whether two OkCupid users successfully matched

Even if such data existed, it would be:
- **Biased**: Only observing pairs that chose to interact
- **Noisy**: Self-reported satisfaction is subjective
- **Domain-specific**: Dating compatibility ≠ friendship compatibility

Therefore, we adopt a **pseudo-labeling** approach where labels are derived from theoretically motivated similarity functions.

---

## 3. Modeling Approach

### 3.1 Pairwise Modeling

Compatibility is a property of **pairs**, not individuals. Our models take (Person A, Person B) as input and output a single compatibility score.

**Pairwise Feature Types**:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| Absolute Difference | \|A - B\| | Captures dissimilarity |
| Element-wise Product | A × B | Captures interaction effects |
| Mean | (A + B) / 2 | Captures joint level |
| Cosine Similarity | cos(A, B) | Captures directional alignment |

For two persons with d-dimensional feature vectors, pairwise features expand to:
- 3d features (diff, product, mean) + similarity scores

### 3.2 Pseudo-Label Generation

Since true compatibility labels don't exist, we generate pseudo-labels based on similarity functions.

**Personality Model**:
```
sim_ocean = (cosine(OCEAN_A, OCEAN_B) + 1) / 2      # [0, 1]
sim_raw = (cosine(raw50_A, raw50_B) + 1) / 2        # [0, 1]
pseudo_label = 0.7 × sim_ocean + 0.3 × sim_raw + ε
pseudo_label = clip(pseudo_label, 0, 1)
```

Where ε ~ N(0, 0.08²) prevents trivial identity mapping.

**Interests Model**:
```
sim_text = cosine(tfidf_A, tfidf_B)                  # [0, 1]
sim_cat = (cosine(cat_A, cat_B) + 1) / 2            # [0, 1]
sim_num = (cosine(num_A, num_B) + 1) / 2            # [0, 1]
pseudo_label = 0.5 × sim_text + 0.3 × sim_cat + 0.2 × sim_num + ε
pseudo_label = clip(pseudo_label, 0, 1)
```

**Theoretical Justification**:
- Personality similarity correlates with relationship satisfaction (Watson et al., 2004)
- Shared interests predict interaction frequency (McPherson et al., 2001)
- The noise term ensures the model learns generalizable patterns, not exact similarity replication

### 3.3 Model Architecture

Both models use **HistGradientBoostingRegressor** from scikit-learn:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_iter | 200 | Sufficient for convergence |
| max_depth | 8 | Captures non-linear interactions |
| learning_rate | 0.1 | Standard default |
| min_samples_leaf | 20 | Prevents overfitting |
| l2_regularization | 0.1 | Regularization |
| early_stopping | True | Prevents overfitting |

**Why Gradient Boosting?**
- Handles mixed feature types naturally
- Provides feature importance for interpretability
- Fast training on tabular data
- No hyperparameter tuning required for baseline

### 3.4 Training Configuration

| Setting | Value |
|---------|-------|
| Max pairs per dataset | 200,000 |
| Train/validation split | 80/20 |
| Seeds for stability | [11, 22, 33, 44, 55] |

---

## 4. Late Fusion Architecture

### 4.1 Why Fusion at Score Level?

The IPIP and OkCupid datasets **do not share user identifiers**. We cannot:
- Join records at the person level
- Train a single model on combined features
- Perform feature-level fusion

Therefore, we adopt **late fusion**: each dataset produces an independent model, and scores are combined post-prediction.

### 4.2 Fusion Formula

```
final_score = α × score_personality + (1 - α) × score_interests
```

Where α = 0.5 (equal weighting) by default.

### 4.3 Empirical Justification

Stability analysis revealed asymmetric behavior:

| Model | Score Mean | Score Std | Feature Stability |
|-------|------------|-----------|-------------------|
| Personality | 0.52 | 0.16 | High (OCEAN cosine dominates) |
| Interests | 0.51 | 0.11 | Medium (text similarity dominates) |

Equal weighting (α = 0.5) was chosen because:
1. Neither model showed clear superiority in stability
2. They capture complementary signals (traits vs. interests)
3. Simple fusion avoids overfitting to stability artifacts

---

## 5. Stability Analysis

### 5.1 Multi-Seed Evaluation

Training was repeated with 5 random seeds: [11, 22, 33, 44, 55]. Each seed affects:
- Pair sampling
- Train/validation split
- Model initialization

### 5.2 Feature-Level Instability

Initial analysis revealed concerning instability at the raw feature level:

| Metric | Personality | Interests |
|--------|-------------|-----------|
| Top-20 Jaccard Overlap | 12.5% | 16.7% |
| Stable Features (all seeds) | 2 | 4 |

This means: **the same feature rarely appeared in top-20 across all 5 seeds**.

### 5.3 Why Construct-Level Aggregation Was Required

Raw feature instability does not imply model instability. Features within the same **construct** (e.g., all Extraversion items) may substitute for each other across seeds.

We aggregated features into interpretable constructs:

**Personality Constructs**:
- 5 OCEAN dimensions + Global Similarity (cosine)

**Interests Constructs**:
- 3 Similarity measures (numeric, categorical, text)
- 13 Categorical constructs (religion, education, lifestyle, etc.)

### 5.4 Construct Stability Results

**Personality** (all stable):

| Construct | Avg Rank | Top-5 Frequency |
|-----------|----------|-----------------|
| Global Similarity | 1.0 | 5/5 |
| Agreeableness | 3.6 | 5/5 |
| Extraversion | 3.6 | 5/5 |
| Conscientiousness | 4.0 | 3/5 |
| Openness | 4.4 | 4/5 |
| Neuroticism | 4.4 | 3/5 |

**Interests** (mixed stability):

| Construct | Avg Rank | Top-5 Frequency | Askable? |
|-----------|----------|-----------------|----------|
| Numeric Similarity | 1.0 | 5/5 | No (computed) |
| Text Interests | 2.6 | 5/5 | Yes (free-text) |
| Categorical Similarity | 2.4 | 5/5 | No (computed) |
| Religion | 4.8 | 4/5 | Yes |
| Job/Career | 6.8 | 1/5 | Excluded |
| Lifestyle Habits | 7.6 | 2/5 | Yes (merged) |
| Education | 7.6 | 0/5 | Yes |
| Family & Relationship | 8.2 | 1/5 | Yes (merged) |

**Key Finding**: Global/aggregate similarity features dominate both models. Individual categorical constructs show lower stability in the interests model.

---

## 6. UI Questionnaire Design

### 6.1 Design Principles

1. **Stability-driven**: Only include constructs with demonstrated cross-seed stability
2. **Construct-level mapping**: Questions map to constructs, not raw features
3. **Minimal burden**: 10 questions total (5 personality + 5 interests)
4. **Interpretability**: Users understand what each question measures

### 6.2 From Features to Questions

**Personality Path**:
```
50 raw items → 5 OCEAN dimensions → 5 representative questions
```

Each OCEAN dimension is represented by one Likert question:

| # | Construct | Question | Stability |
|---|-----------|----------|-----------|
| P1 | Extraversion | "I feel comfortable around people" | 5/5 |
| P2 | Agreeableness | "I am interested in other people's problems" | 5/5 |
| P3 | Conscientiousness | "I pay attention to details" | 3/5 |
| P4 | Openness | "I have a vivid imagination" | 4/5 |
| P5 | Neuroticism | "I get stressed out easily" | 3/5 |

**Interests Path**:
```
100+ features → 16 constructs → 5 UI items (3 categorical + 1 merged + 1 free-text)
```

| # | Construct | Format | Stability | Rationale |
|---|-----------|--------|-----------|-----------|
| I1 | Religion | Categorical | 4/5 | Only stable categorical |
| I2 | About Me | Free-text | 5/5 | Captures text similarity |
| I3 | Lifestyle | Categorical (merged) | 2/5 | Drinks + smokes + drugs |
| I4 | Family | Categorical (merged) | 1/5 | Status + offspring |
| I5 | Education | Categorical | 0/5 | Selected over Job |

### 6.3 Excluded Constructs

| Construct | Reason |
|-----------|--------|
| Zodiac Sign | No predictive value (excluded by design) |
| Job/Career | Low stability (1/5), sensitive to career stage |
| Body Type | Low stability (0/5), sensitive topic |
| Diet | Low stability (1/5) |
| Pets | Low stability (0/5) |
| Height, Income | Low stability, typically implicit |
| Sex, Orientation | Typically captured in profile setup |

### 6.4 Handling Low-Stability Constructs

Where categorical stability was weak, we adopted two strategies:

1. **Merging**: Combined related low-stability fields into single questions
   - Lifestyle = drinks + smokes + drugs
   - Family = status + offspring

2. **Free-text substitution**: The "About Me" field captures text similarity (rank 2.6, 5/5 stability), which subsumes many categorical constructs that users would naturally mention.

---

## 7. Inference & UI Implementation

### 7.1 End-to-End Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Inference API   │───▶│  Trained Models │
│  (10 questions) │    │  (predict.py)    │    │  (seed_11)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                      │                       │
        ▼                      ▼                       ▼
   User Inputs            Feature Transform         Score Output
   - 5 Likert             - OCEAN scaling           - personality_score
   - 4 categorical        - One-hot encoding        - interests_score
   - 1 free-text          - TF-IDF transform        - final_score
```

### 7.2 Feature Space Mapping

The UI collects fewer features than the training data. We handle this by:

**Personality**:
- UI provides 5 OCEAN scores (one per dimension)
- Raw 50-item features set to neutral (3.0) for compatibility
- OCEAN cosine similarity computed from UI inputs

**Interests**:
- UI provides subset of categoricals (religion, education, lifestyle, family)
- Missing categoricals filled with "unknown"
- Age/height/income set to training medians
- TF-IDF computed from "About Me" text

### 7.3 Validation Checks

| Check | Implementation |
|-------|----------------|
| Likert range | Slider constrained to 1-5 |
| Enum validity | Dropdown with fixed options |
| Text length | Warning if < 200 chars, error if empty |
| Score bounds | Clipped to [0, 1] |
| Symmetry | predict(A,B) == predict(B,A) verified |

### 7.4 Confidence Indicator

The UI displays a confidence level based on input completeness:

| Level | Criteria |
|-------|----------|
| High | Both texts ≥ 200 chars, no "unknown" selections |
| Medium | Texts ≥ 100 chars, ≤ 2 "unknown" selections |
| Low | Short texts or many "unknown" selections |

---

## 8. Limitations

### 8.1 No Ground Truth

**The most fundamental limitation**: We cannot evaluate real-world predictive accuracy. All metrics are based on:
- Stability across random seeds
- Internal consistency
- Theoretical plausibility

We explicitly do NOT claim that high scores predict successful relationships.

### 8.2 Text Length Sensitivity

The interests model relies heavily on text similarity (text_cosine_sim). Short inputs produce:
- Sparse TF-IDF vectors
- Low similarity variance
- Scores clustered around 0.47-0.48

**Mitigation**: UI requires minimum 200 characters and displays warning for shorter inputs.

### 8.3 Population Bias

Both datasets have known biases:
- **IPIP**: Skewed toward English-speaking, internet-accessible populations
- **OkCupid**: San Francisco area, 2011-2012, specific age demographics

The model may not generalize to different populations.

### 8.4 Categorical Coverage Gap

The UI captures only 5 of 14 original OkCupid categorical fields. Excluded fields are set to "unknown", which:
- Reduces discriminative power
- May bias scores toward users who would have selected "unknown" anyway

### 8.5 Single-Question OCEAN Measurement

The UI uses 1 question per OCEAN dimension; the original IPIP uses 10. This trades:
- **Gained**: Lower user burden, faster completion
- **Lost**: Measurement reliability, nuanced dimension capture

### 8.6 Static Fusion Weight

Alpha = 0.5 is fixed. In reality:
- Some users may have richer text profiles (should weight interests higher)
- Some users may have distinctive personality profiles (should weight personality higher)

---

## 9. Future Work

### 9.1 Better Text Embeddings

Replace TF-IDF with:
- **Sentence transformers** (e.g., all-MiniLM-L6-v2)
- **Domain-specific embeddings** trained on dating/social profiles

Expected improvements:
- Better semantic similarity for short texts
- Reduced vocabulary dependence
- Cross-lingual potential

### 9.2 Adaptive Fusion Weights

Learn α dynamically based on:
- Input completeness (text length, non-unknown count)
- Prediction confidence (distance from 0.5)
- User-specific calibration

### 9.3 Asymmetric Compatibility

Current model: similarity(A, B) = similarity(B, A)

Real compatibility may be asymmetric:
- A may be more attracted to B than vice versa
- Power dynamics, attachment styles, etc.

Future work could model directed compatibility.

### 9.4 Temporal Dynamics

Personality and interests change over time. Future work could:
- Track profile changes
- Model compatibility trajectory
- Predict long-term vs. short-term compatibility

### 9.5 Explainability

Current output: single score.

Users want to know:
- "Why did we score 65%?"
- "What do we have in common?"
- "Where do we differ?"

Future work: generate natural language explanations from feature contributions.

### 9.6 A/B Testing Framework

If deployed, implement:
- Randomized score variations
- Outcome tracking (did users interact?)
- Causal inference on score impact

This would provide the first ground-truth evaluation, though with selection bias caveats.

---

## Appendix: Artifacts

### A.1 Directory Structure

```
artifacts/
├── runs/
│   ├── seed_11/
│   │   ├── models/
│   │   │   ├── model_personality.joblib
│   │   │   └── model_interests.joblib
│   │   ├── preprocessors/
│   │   │   ├── preprocessor_personality.joblib
│   │   │   └── preprocessor_interests.joblib
│   │   ├── reports/
│   │   │   ├── feature_importance_personality.csv
│   │   │   ├── feature_importance_interests.csv
│   │   │   ├── evaluation_personality.json
│   │   │   └── evaluation_interests.json
│   │   └── configs/
│   │       ├── config_used.yaml
│   │       ├── fusion_config.json
│   │       └── pseudo_label_config.json
│   ├── seed_22/
│   ├── seed_33/
│   ├── seed_44/
│   └── seed_55/
├── stability/
│   ├── stability_summary.json
│   └── feature_rank_stability.csv
└── constructs/
    ├── personality_construct_stability.csv
    ├── interests_construct_stability.csv
    ├── construct_analysis.json
    └── ui_questionnaire_proposal.md
```

### A.2 Key Files

| File | Purpose |
|------|---------|
| `pipeline/run.py` | Training pipeline entry point |
| `pipeline/inference/predict.py` | Inference API |
| `pipeline/inference/schema.py` | Input data structures |
| `ui/app.py` | Streamlit web interface |
| `configs/config.yaml` | Training configuration |
| `configs/ipip50_mapping.yaml` | OCEAN dimension mapping |

### A.3 Reproducibility

To reproduce training:
```bash
python -m pipeline.run --config configs/config.yaml --seed 11 --output-dir artifacts/runs/seed_11
```

To run inference tests:
```bash
python scripts/test_inference.py
```

To launch UI:
```bash
streamlit run ui/app.py
```

---

## References

1. Goldberg, L. R. (1992). The development of markers for the Big-Five factor structure. *Psychological Assessment*, 4(1), 26-42.

2. Watson, D., Klohnen, E. C., Srivastava, S., et al. (2004). Match makers and deal breakers: Analyses of assortative mating in newlywed couples. *Journal of Personality*, 72(5), 1029-1068.

3. McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a feather: Homophily in social networks. *Annual Review of Sociology*, 27(1), 415-444.

4. IPIP. (2019). International Personality Item Pool. https://ipip.ori.org/

---

**End of Report**
