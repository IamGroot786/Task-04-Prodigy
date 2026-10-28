# Dataset source and attribution

**Primary dataset:** Twitter Entity Sentiment Analysis sample supplied for Prodigy InfoTech Data Science Task 4.

- Official folder: https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204
- Training CSV: https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%204/twitter_training.csv
- Verified source file SHA (Git blob): `002662034fca0008eeff5fe09046ba778b388066`
- Retrieved source payload size: 10,288,167 bytes
- Expected four headerless columns: tweet ID, entity/topic, sentiment label, tweet text.
- Source labels: Positive, Negative, Neutral, Irrelevant.

## Dataset selection (26 October 2026)

The connected GitHub account `IamGroot786` was reviewed. No separate social-media sentiment dataset was found in the accessible user repositories. The Task 3 README was fetched directly and confirms that Task 3 uses the UCI/Prodigy Bank Marketing dataset (`bank-full.csv`); that dataset is not reused. Task 2's Titanic data is also excluded. Therefore, this project uses the official Prodigy Task 4 sample as requested.

The official Task 4 path successfully returned `twitter_training.csv`; a request for `twitter_validation.csv` at the same path returned not found. The project therefore analyzes the available training file and supports a validation CSV only as an optional local addition, without claiming a train/validation comparison for this run.

## License and handling note

The raw CSV is not included in this ZIP. The project includes derived aggregate counts and visualizations. Confirm the upstream repository's current license and attribution requirements before redistributing the raw dataset.
