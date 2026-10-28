import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyze_sentiment as analysis  # noqa: E402


class SentimentAnalysisTests(unittest.TestCase):
    def test_normalize_all_source_sentiments(self):
        self.assertEqual(analysis.normalize_sentiment(" Positive  "), "Positive")
        self.assertEqual(analysis.normalize_sentiment("NEGATIVE"), "Negative")
        self.assertEqual(analysis.normalize_sentiment("neutral"), "Neutral")
        self.assertEqual(analysis.normalize_sentiment("IrRelevant"), "Irrelevant")
        self.assertIsNone(analysis.normalize_sentiment("mixed"))
        self.assertIsNone(analysis.normalize_sentiment(None))

    def test_headerless_source_columns_and_cleaning(self):
        sample = pd.DataFrame({
            "tweet_id": [1, 2, 3, 4, 5],
            "entity": ["Xbox", "Xbox", "PlayStation", "PlayStation", "Nintendo"],
            "sentiment": ["Positive", "Negative", "Neutral", "Irrelevant", "bad-label"],
            "text": ["Love this #Xbox!", "Terrible update", "News today", "", "unknown label"],
            "source_split": ["training", "training", "validation", "validation", "training"],
        })
        cleaned, audit = analysis.clean_dataframe(sample)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(set(cleaned["sentiment"]), {"Positive", "Negative", "Neutral"})
        self.assertEqual(audit["invalid_sentiment_label_rows_before_cleaning"], 1)
        self.assertEqual(audit["missing_or_empty_text_rows_before_cleaning"], 1)
        self.assertIn("text_clean", cleaned.columns)
        self.assertTrue(cleaned.loc[cleaned["entity"] == "Xbox", "text_clean"].iloc[0].find("xbox") >= 0)

    def test_duplicates_removed_by_entity_label_and_text(self):
        sample = pd.DataFrame({
            "tweet_id": [1, 2, 3], "entity": ["Xbox", "Xbox", "Xbox"],
            "sentiment": ["Positive", "Positive", "Negative"],
            "text": ["Same post", "Same post", "Same post"],
        })
        cleaned, audit = analysis.clean_dataframe(sample)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(audit["duplicates_removed"], 1)

    def test_missing_required_columns_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "entity, sentiment, text"):
            analysis.clean_dataframe(pd.DataFrame({"platform": ["Twitter"]}))

    def test_end_to_end_outputs_with_small_fixture(self):
        sample = pd.DataFrame({
            "tweet_id": range(1, 9),
            "entity": ["Xbox", "Xbox", "Xbox", "PlayStation", "PlayStation", "Nintendo", "Nintendo", "Nintendo"],
            "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Irrelevant", "Positive", "Negative", "Neutral"],
            "text": ["Love the new game", "Awful update", "A game announcement", "Great console", "Not about topic", "Amazing release", "Terrible lag", "New release today"],
            "source_split": ["training"] * 4 + ["validation"] * 4,
        })
        cleaned, audit = analysis.clean_dataframe(sample)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with patch.object(analysis, "CLEAN_PATH", temp / "processed.csv"), \
                 patch.object(analysis, "FIGURES", temp / "figures"), \
                 patch.object(analysis, "REPORTS", temp / "reports"):
                report = analysis.make_outputs(cleaned, audit)
                self.assertEqual(report["rows_analyzed"], 8)
                self.assertEqual(report["source_file"], "twitter_training.csv + twitter_validation.csv")
                self.assertEqual(sum(report["sentiment_counts"].values()), 8)
                self.assertTrue((temp / "figures" / "sentiment_distribution.png").exists())
                self.assertTrue((temp / "figures" / "sentiment_by_source_split.png").exists())
                self.assertTrue((temp / "figures" / "sentiment_by_entity.png").exists())
                self.assertTrue((temp / "figures" / "positive_negative_balance_by_entity.png").exists())
                self.assertTrue((temp / "reports" / "sentiment_by_source_split_percentage.csv").exists())
                self.assertTrue((temp / "reports" / "executive_summary.md").exists())
                summary = json.loads((temp / "reports" / "analysis_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["rows_analyzed"], 8)
                self.assertIn("Irrelevant", summary["sentiment_counts"])
                self.assertEqual(sum(summary["sentiment_counts"].values()), 8)
                self.assertEqual(sum(summary["source_split_counts"].values()), 8)
                for percentages in summary["source_split_sentiment_percentages"].values():
                    self.assertAlmostEqual(sum(percentages.values()), 100.0, delta=0.1)


    def test_training_only_summary_does_not_claim_validation_file(self):
        sample = pd.DataFrame({
            "tweet_id": [1, 2],
            "entity": ["Xbox", "Xbox"],
            "sentiment": ["Positive", "Negative"],
            "text": ["Great release", "Bad update"],
            "source_split": ["training", "training"],
        })
        cleaned, audit = analysis.clean_dataframe(sample)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with patch.object(analysis, "CLEAN_PATH", temp / "processed.csv"), \
                 patch.object(analysis, "FIGURES", temp / "figures"), \
                 patch.object(analysis, "REPORTS", temp / "reports"), \
                 patch.object(analysis, "TRAIN_PATH", temp / "missing_training.csv"):
                report = analysis.make_outputs(cleaned, audit)
                self.assertEqual(report["source_file"], "twitter_training.csv")
                self.assertNotIn("twitter_validation.csv", report["dataset_source"])
                self.assertIn("Positive", report["sentiment_counts"])

    def test_empty_cleaned_dataset_is_not_reported_as_success(self):
        with self.assertRaisesRegex(ValueError, "No usable records"):
            analysis.make_outputs(pd.DataFrame(), {})


if __name__ == "__main__":
    unittest.main()
