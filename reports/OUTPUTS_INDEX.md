# Generated outputs (quality-reviewed 28 October 2026)

- `analysis_summary.json` — source hash, cleaning counts, sentiment percentages, entity summary, and limitations.
- `cleaning_audit.json` — raw/retained row counts and removal totals.
- `sentiment_distribution.csv` — full-file counts and percentages.
- `entity_sentiment_summary.csv` — per-entity counts and positive/negative balance for all 32 entities.
- `sentiment_by_entity_percentage.csv` — normalized sentiment mix for the 15 highest-volume entities.
- `sentiment_by_source_split_percentage.csv` — sentiment percentages by available source split.
- `term_frequency_by_sentiment.csv` — frequent terms per source sentiment label.
- `executive_summary.md` — concise evidence-based results and cautions.
- `outputs/figures/*.png` — charts generated from the full-file aggregate results.

The raw source CSV is intentionally not included in this ZIP.

## Final package quality checks (28 October 2026)

- 7 automated unit tests pass against synthetic fixtures.
- Python modules compile successfully.
- Packaged PNG files are valid and readable.
- Aggregate sentiment counts reconcile to 70,522 retained records.
- The final ZIP is a local submission package; GitHub upload is not confirmed.
- The source-split CSV has one row (`training`) because no validation CSV was available; it is not a train/validation comparison.
