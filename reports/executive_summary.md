# Executive Summary — Twitter Entity Sentiment Analysis

**Source rows:** 74,682
**Rows retained after cleaning:** 70,522
**Entities/topics:** 32

## Sentiment distribution

- **Negative:** 21,462 (30.43%)
- **Positive:** 19,412 (27.53%)
- **Neutral:** 17,271 (24.49%)
- **Irrelevant:** 12,377 (17.55%)

## Key entity comparisons

- **Most positive balance:** AssassinsCreed (+46.88 percentage points; 63.91% Positive vs 17.03% Negative, n=2,131).
- **Most negative balance:** MaddenNFL (-55.58 percentage points; 71.86% Negative vs 16.28% Positive, n=2,285).
- **NBA2K** is also predominantly negative in the sample: 63.09% Negative (n=2,284).

## Cleaning audit

- 858 rows had missing or empty tweet text and were removed.
- 3,302 duplicate entity-label-text rows were removed.
- No invalid sentiment labels were found.

## Interpretation cautions

- Source labels are inherited annotations and were not independently verified.
- This dataset is not a probability sample of all social-media users.
- Irrelevant is kept separate from Neutral.
- The validation CSV was not found at the official sample path, so this run does not report validation split comparisons.
- Frequent terms are descriptive and can be affected by abbreviations, platform language, and context.
