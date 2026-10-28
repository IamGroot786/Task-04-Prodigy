"""Clean, analyze, and visualize the Prodigy Task 4 Twitter sentiment dataset."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "twitter_training.csv"
VALIDATION_PATH = RAW_DIR / "twitter_validation.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "twitter_sentiment_cleaned.csv"
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "reports"
VALID_SENTIMENTS = {
    "positive": "Positive", "negative": "Negative", "neutral": "Neutral", "irrelevant": "Irrelevant"
}
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative", "Irrelevant"]
STOPWORDS = set("""a an and are as at be been being but by can could did do does doing for from had has have he her hers him his how i if in into is it its just me my of on or our ours she so than that the their them there they this to too us was we were what when where which who will with would you your yours rt http https www user url game games playing play get got im ive its one really like love hate good bad great terrible today still even much more most some not no yes""".split())


def normalize_sentiment(value: object) -> str | None:
    if pd.isna(value):
        return None
    return VALID_SENTIMENTS.get(str(value).strip().lower())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    aliases = {"tweet": "text", "tweet_text": "text", "tweet_content": "text", "topic": "entity", "brand": "entity", "id": "tweet_id"}
    out = out.rename(columns={c: aliases[c] for c in out.columns if c in aliases})
    out = out.loc[:, ~out.columns.str.match(r"^unnamed(:_?\d+)?$")]
    return out


def load_source_files() -> pd.DataFrame:
    """Load the official training CSV and any optional validation CSV available locally.

    The official Task 4 folder currently exposes twitter_training.csv. If a separate
    validation file is supplied later, it is included and marked as its own split.
    """
    frames = []
    for path, split in [(TRAIN_PATH, "training"), (VALIDATION_PATH, "validation")]:
        if not path.exists():
            if split == "training":
                raise FileNotFoundError(f"Missing {path.name}. Run `python src/fetch_data.py` first.")
            continue
        frame = pd.read_csv(
            path, header=None, names=["tweet_id", "entity", "sentiment", "text"],
            encoding="utf-8-sig", dtype={"tweet_id": "string", "entity": "string", "sentiment": "string", "text": "string"},
            on_bad_lines="warn", low_memory=False,
        )
        frame["source_split"] = split
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No source CSVs found. Run `python src/fetch_data.py` first.")
    return pd.concat(frames, ignore_index=True)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize labels, remove unusable records, deduplicate, and return a cleaning audit."""
    out = standardize_columns(df)
    required = {"entity", "sentiment", "text"}
    if not required.issubset(out.columns):
        missing = sorted(required - set(out.columns))
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")
    if "tweet_id" not in out.columns:
        out["tweet_id"] = pd.NA
    if "source_split" not in out.columns:
        out["source_split"] = "unspecified"

    before_rows = int(len(out))
    missing_before = {str(k): int(v) for k, v in out.isna().sum().items()}
    original_labels = out["sentiment"].astype("string").str.strip().str.lower()
    recognized = original_labels.isin(VALID_SENTIMENTS)
    invalid_label_rows = int((~recognized & original_labels.notna() & original_labels.ne("")).sum())
    missing_text_rows = int((out["text"].isna() | out["text"].astype("string").str.strip().eq("")).sum())

    for col in ["tweet_id", "entity", "text", "source_split"]:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    out["sentiment"] = out["sentiment"].map(normalize_sentiment)
    out = out.dropna(subset=["text", "sentiment"])
    out = out[out["text"].str.len().fillna(0) > 0]
    out["entity"] = out["entity"].fillna("Unknown entity")
    out["source_split"] = out["source_split"].fillna("unspecified")

    before_dedup = int(len(out))
    # Tweet IDs alone are not used as the key because IDs can be repeated in different entity contexts.
    out = out.drop_duplicates(subset=["entity", "sentiment", "text"], keep="first")
    duplicates_removed = before_dedup - int(len(out))
    out = out.reset_index(drop=True)
    out["text_clean"] = (
        out["text"].astype(str)
        .str.replace(r"https?://\S+|www\.\S+|(?:pic\.)?twitter\.com/\S+|t\.co/\S+", " ", regex=True)
        .str.replace(r"@[\w]+", " ", regex=True)
        .str.replace(r"#[\w]+", lambda m: m.group(0)[1:], regex=True)
        .str.replace(r"[^\w\s']", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    )
    audit = {
        "rows_before": before_rows,
        "rows_after": int(len(out)),
        "rows_removed_total": int(before_rows - len(out)),
        "invalid_sentiment_label_rows_before_cleaning": invalid_label_rows,
        "missing_or_empty_text_rows_before_cleaning": missing_text_rows,
        "duplicates_removed": int(duplicates_removed),
        "missing_values_before": missing_before,
        "missing_values_after": {str(k): int(v) for k, v in out.isna().sum().items()},
        "sentiment_labels_retained": sorted(out["sentiment"].dropna().unique().tolist()),
        "rows_by_source_split_after_cleaning": {str(k): int(v) for k, v in out["source_split"].value_counts().items()},
    }
    return out, audit


def _save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path,
              rotate: int = 0, horizontal: bool = False) -> None:
    if series.empty:
        return
    height = max(4, min(10, 0.36 * len(series))) if horizontal else 5.5
    fig, ax = plt.subplots(figsize=(11, height))
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax)
    else:
        series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate:
        ax.tick_params(axis="x", labelrotation=rotate)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _term_counts(texts: pd.Series) -> Counter:
    words: list[str] = []
    for text in texts.dropna().astype(str):
        tokens = re.findall(r"[a-z][a-z']{1,}", text.lower())
        words.extend(token for token in tokens if token not in STOPWORDS and not token.startswith("user"))
    return Counter(words)


def make_outputs(df: pd.DataFrame, audit: dict) -> dict:
    if df.empty:
        raise ValueError("No usable records remain after cleaning; refusing to generate empty findings.")
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    sentiment_counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    sentiment_pct = (sentiment_counts / len(df) * 100).round(2)
    sentiment_table = pd.DataFrame({"count": sentiment_counts, "percentage": sentiment_pct})
    sentiment_table.to_csv(REPORTS / "sentiment_distribution.csv")
    _save_bar(sentiment_counts, "Overall Tweet Sentiment Distribution", "Sentiment label", "Tweets", FIGURES / "sentiment_distribution.png")

    # Keep the official training/validation origin visible so combined-source
    # percentages can be checked for major composition differences.
    split_sentiment = pd.crosstab(df["source_split"], df["sentiment"], normalize="index") * 100
    split_sentiment = split_sentiment.reindex(columns=SENTIMENT_ORDER, fill_value=0).round(2)
    split_sentiment.to_csv(REPORTS / "sentiment_by_source_split_percentage.csv")
    if not split_sentiment.empty and df["source_split"].nunique() > 1:
        ax = split_sentiment.plot(kind="bar", figsize=(9, 5))
        ax.set_title("Sentiment Mix by Source Split")
        ax.set_xlabel("Source split")
        ax.set_ylabel("Share of labeled tweets (%)")
        ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES / "sentiment_by_source_split.png", dpi=160, bbox_inches="tight")
        plt.close()

    entity_counts = df["entity"].value_counts()
    top_entities = entity_counts.head(15).index.tolist()
    _save_bar(entity_counts.head(15), "Top 15 Topics / Entities by Tweet Volume", "Entity", "Tweets", FIGURES / "top_entities_by_volume.png", rotate=55)
    entity_sentiment = pd.crosstab(df.loc[df["entity"].isin(top_entities), "entity"],
                                   df.loc[df["entity"].isin(top_entities), "sentiment"], normalize="index") * 100
    entity_sentiment = entity_sentiment.reindex(columns=SENTIMENT_ORDER, fill_value=0).round(2)
    entity_sentiment.to_csv(REPORTS / "sentiment_by_entity_percentage.csv")
    if not entity_sentiment.empty:
        ax = entity_sentiment.plot(kind="bar", stacked=True, figsize=(13, 6))
        ax.set_title("Sentiment Composition by Top 15 Entities")
        ax.set_xlabel("Entity / topic")
        ax.set_ylabel("Share of labeled tweets (%)")
        ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=50, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES / "sentiment_by_entity.png", dpi=160, bbox_inches="tight")
        plt.close()

    entity_summary = pd.crosstab(df["entity"], df["sentiment"]).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    entity_summary["total_tweets"] = entity_summary.sum(axis=1)
    entity_summary["positive_pct"] = (entity_summary["Positive"] / entity_summary["total_tweets"] * 100).round(2)
    entity_summary["negative_pct"] = (entity_summary["Negative"] / entity_summary["total_tweets"] * 100).round(2)
    entity_summary["positive_minus_negative_pp"] = (entity_summary["positive_pct"] - entity_summary["negative_pct"]).round(2)
    entity_summary = entity_summary.sort_values("total_tweets", ascending=False)
    entity_summary.to_csv(REPORTS / "entity_sentiment_summary.csv")
    balance = entity_summary.head(15)["positive_minus_negative_pp"]
    _save_bar(balance, "Positive Minus Negative Share by Entity (Top 15 by Volume)", "Positive share − negative share (percentage points)", "Percentage points", FIGURES / "positive_negative_balance_by_entity.png", rotate=50, horizontal=True)

    term_rows = []
    for sentiment in SENTIMENT_ORDER:
        counts = _term_counts(df.loc[df["sentiment"] == sentiment, "text_clean"])
        for term, count in counts.most_common(20):
            term_rows.append({"sentiment": sentiment, "term": term, "count": int(count)})
        if counts:
            _save_bar(pd.Series(dict(counts.most_common(15)), dtype="int64"),
                      f"Most Frequent Terms — {sentiment}", "Term", "Occurrences",
                      FIGURES / f"top_terms_{sentiment.lower()}.png", rotate=45)
    pd.DataFrame(term_rows, columns=["sentiment", "term", "count"]).to_csv(REPORTS / "term_frequency_by_sentiment.csv", index=False)

    entity_share = (entity_counts / len(df) * 100).round(2)
    source_splits = set(df["source_split"].dropna().astype(str).tolist())
    source_file_names = []
    if "training" in source_splits:
        source_file_names.append("twitter_training.csv")
    if "validation" in source_splits:
        source_file_names.append("twitter_validation.csv")
    if not source_file_names:
        source_file_names.append("locally supplied CSV")
    source_file_label = " + ".join(source_file_names)
    source_git_blob_sha = None
    source_bytes = None
    if TRAIN_PATH.exists():
        source_payload = TRAIN_PATH.read_bytes()
        source_bytes = len(source_payload)
        source_git_blob_sha = hashlib.sha1(
            b"blob " + str(len(source_payload)).encode("ascii") + b"\0" + source_payload
        ).hexdigest()

    eligible_entities = entity_summary[entity_summary["total_tweets"] >= 30]
    key_findings = [
        f"{sentiment_counts.idxmax()} is the largest retained source-label class at "
        f"{float(sentiment_pct[sentiment_counts.idxmax()]):.2f}%; "
        f"Positive is {float(sentiment_pct.get('Positive', 0)):.2f}%."
    ]
    if not eligible_entities.empty:
        most_positive = eligible_entities.sort_values("positive_minus_negative_pp", ascending=False).iloc[0]
        most_negative = eligible_entities.sort_values("positive_minus_negative_pp", ascending=True).iloc[0]
        key_findings.append(
            f"{most_positive.name} has the largest positive-minus-negative balance among entities "
            f"with at least 30 retained records ({most_positive['positive_minus_negative_pp']:+.2f} percentage points)."
        )
        key_findings.append(
            f"{most_negative.name} has the most negative balance among entities with at least 30 retained records "
            f"({most_negative['positive_minus_negative_pp']:+.2f} percentage points); "
            f"{most_negative['negative_pct']:.2f}% of its retained records are labeled Negative."
        )

    report = {
        "project": "Prodigy Infotech Task 4 — Twitter Entity Sentiment Analysis",
        "source_file": source_file_label,
        "source_url": "https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204",
        "source_git_blob_sha": source_git_blob_sha,
        "source_bytes": source_bytes,
        "raw_rows": int(audit.get("rows_before", len(df))),
        "dataset_source": f"Prodigy InfoTech Task 4 sample ({source_file_label})",
        "key_findings": key_findings,
        "rows_analyzed": int(len(df)),
        "entities_analyzed": int(df["entity"].nunique()),
        "sentiment_counts": {str(k): int(v) for k, v in sentiment_counts.items()},
        "sentiment_percentages": {str(k): float(v) for k, v in sentiment_pct.items()},
        "source_split_counts": {str(k): int(v) for k, v in df["source_split"].value_counts().items()},
        "source_split_sentiment_percentages": {
            str(split): {str(label): float(value) for label, value in row.items()}
            for split, row in split_sentiment.iterrows()
        },
        "top_entities_by_volume": [{"entity": str(k), "count": int(v), "percentage_of_all_rows": float(entity_share[k])} for k, v in entity_counts.head(15).items()],
        "entity_sentiment_summary": [
            {"entity": str(idx), **{str(col): (int(val) if col in SENTIMENT_ORDER or col == "total_tweets" else float(val)) for col, val in row.items()}}
            for idx, row in entity_summary.head(15).iterrows()
        ],
        "cleaning_audit": audit,
        "limitations": [
            "Source sentiment labels are inherited annotations; they are not independently verified by this project.",
            "Irrelevant is retained as a fourth source class and is not merged into Neutral.",
            "This is a labeled dataset sample, not a probability sample of all social-media users; results should not be generalized to the public as a whole.",
            "Word-frequency charts are descriptive and do not establish causation or explain the context of each tweet.",
            "Entity-wise proportions should be read alongside tweet counts because small groups can have unstable percentages.",
        ],
    }
    (REPORTS / "analysis_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Executive Summary — Twitter Entity Sentiment Analysis", "",
        f"**Tweets analyzed after cleaning:** {len(df):,}",
        f"**Distinct entities/topics:** {df['entity'].nunique():,}", "",
        "## Overall sentiment distribution", "",
    ]
    for label in SENTIMENT_ORDER:
        lines.append(f"- **{label}:** {int(sentiment_counts[label]):,} tweets ({float(sentiment_pct[label]):.2f}%)")
    lines += ["", "## Key entity comparisons", ""]
    lines += [f"- {finding}" for finding in key_findings[1:]]
    lines += ["", "## Cleaning audit", "",
              f"- {audit.get('missing_or_empty_text_rows_before_cleaning', 0):,} rows had missing or empty tweet text and were removed.",
              f"- {audit.get('duplicates_removed', 0):,} duplicate entity-label-text rows were removed.",
              f"- {audit.get('invalid_sentiment_label_rows_before_cleaning', 0):,} rows had invalid sentiment labels before cleaning."]
    lines += ["", "## Interpretation cautions", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "Counts and percentages are generated from the locally available source files.", ""]
    (REPORTS / "executive_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Analyzed {len(df):,} tweets across {df['entity'].nunique():,} entities.")
    print("Sentiment counts:", {k: int(v) for k, v in sentiment_counts.items()})
    print("Saved charts to:", FIGURES)
    print("Saved report to:", REPORTS / "executive_summary.md")
    return report


def main() -> None:
    raw = load_source_files()
    print(f"Raw combined shape: {raw.shape}")
    cleaned, audit = clean_dataframe(raw)
    (REPORTS).mkdir(parents=True, exist_ok=True)
    (REPORTS / "cleaning_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Cleaning audit:", json.dumps(audit, indent=2, ensure_ascii=False))
    make_outputs(cleaned, audit)


if __name__ == "__main__":
    main()
