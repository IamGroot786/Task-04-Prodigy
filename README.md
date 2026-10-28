# Prodigy Infotech — Data Science Task 4
## Twitter Entity Sentiment Analysis

**Project window:** 22–28 October 2026  
**Progress updated:** 27 October 2026  
**Status:** Full-file analysis completed for the available official training CSV; quality checks are in progress. The validation file is optional for reproduction.

## Objective
Analyze and visualize sentiment patterns in social-media posts to compare public reactions across topics, brands, and other entities.

## Dataset decision
The accessible repositories on the connected GitHub account (`IamGroot786`) were reviewed. No suitable alternative social-media sentiment dataset was found. The project uses the official Prodigy Task 4 sample: `twitter_training.csv`. The same folder path did not return `twitter_validation.csv`, so no validation-split comparison is claimed in the current results.

- Official folder: https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204
- Source details and Git blob SHA: `data/SOURCE.md`
- Task 3 is separate: `IamGroot786/Task-03-Prodigy` uses the UCI/Prodigy Bank Marketing `bank-full.csv` dataset. Task 2's Titanic dataset is also excluded.
- Source labels are Positive, Neutral, Negative, and Irrelevant. Irrelevant is kept as its own category.

## Verified full-file results
The official training file contains **74,682 source rows**. Cleaning removed **858 rows with missing/empty tweet text** and **3,302 duplicate entity-label-text records**, retaining **70,522 tweets** across **32 entities**. No unrecognized sentiment labels were found.

| Sentiment | Cleaned tweets | Share |
|---|---:|---:|
| Negative | 21,462 | 30.43% |
| Positive | 19,412 | 27.53% |
| Neutral | 17,271 | 24.49% |
| Irrelevant | 12,377 | 17.55% |

### Selected findings
- Negative is the largest retained class at **30.43%**, followed by Positive at **27.53%**. This is a description of the labeled sample, not a population estimate.
- Among entities with at least 30 retained tweets, **AssassinsCreed** has the largest positive-minus-negative share balance (**+46.88 percentage points**; 63.91% Positive vs 17.03% Negative).
- **MaddenNFL** has the most negative balance (**−55.58 percentage points**; 71.86% Negative vs 16.28% Positive). NBA2K is also strongly negative in this sample (63.09% Negative).
- These are comparisons of inherited dataset labels, not newly predicted sentiment or proof of why users feel that way.

## Workflow
1. Download the official training CSV without altering the source file.
2. Load the four headerless columns and normalize the four source labels.
3. Audit missing text, invalid labels, and duplicates.
4. Compare overall sentiment distribution and sentiment mix by entity.
5. Calculate positive-minus-negative percentage-point balance for each entity.
6. Extract frequent terms per source sentiment class after basic text cleaning.
7. Save cleaned data, charts, summary tables, audit, and an executive summary.

## Setup and run
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows CMD: .venv\Scripts\activate.bat
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/fetch_data.py
python src/analyze_sentiment.py
python -m unittest discover -s tests -v
```

If GitHub is inaccessible, download `twitter_training.csv` from the official folder and place it at `data/raw/twitter_training.csv`. An additional `twitter_validation.csv`, if obtained separately, is automatically included by the loader; the current project results do not depend on it.

## Outputs
- `outputs/figures/sentiment_distribution.png`
- `outputs/figures/top_entities_by_volume.png`
- `outputs/figures/sentiment_by_entity.png`
- `outputs/figures/positive_negative_balance_by_entity.png`
- `outputs/figures/top_terms_*.png`
- `reports/cleaning_audit.json`
- `reports/sentiment_distribution.csv`
- `reports/entity_sentiment_summary.csv`
- `reports/sentiment_by_entity_percentage.csv`
- `reports/term_frequency_by_sentiment.csv`
- `reports/analysis_summary.json`
- `reports/executive_summary.md`

## Final quality check status (28 October 2026)
- Python syntax compilation passed for the analysis, downloader, and test modules.
- Automated unit tests passed locally (7 tests), including a regression check that a training-only run does not claim a validation file.
- The packaged PNG charts were checked for valid image dimensions; the aggregate sentiment counts sum to the 70,522 retained rows.
- The connected GitHub account previously exposed `Task-01-Prodigy`, `Task-02-Prodigy`, and `Task-03-Prodigy`; `Task-04-Prodigy` was not present in the accessible repository list during the latest check. This package is prepared locally; a GitHub upload is not claimed.

## Submission status
- Final ZIP includes source code, notebook, tests, aggregate CSV/JSON reports, eight PNG visualizations, dataset attribution, and progress notes.
- Raw source CSV is intentionally excluded; run the downloader before reproducing the analysis.
- Final package integrity and analysis-pipeline tests were checked on 28 October 2026.

## Limitations
- Source labels are inherited annotations and were not independently verified.
- The dataset is not a probability sample of all social-media users; do not generalize the percentages to the public as a whole.
- Entity labels and class counts are imbalanced; interpret sentiment percentages with sample counts.
- Frequent terms are descriptive and can be affected by platform-specific language, abbreviations, and context.
- The validation CSV was not found at the official sample path, so this run does not report a validation comparison.
- The raw dataset is not bundled; verify upstream terms before redistributing it.
