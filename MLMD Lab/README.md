# ML Metadata Lab — Wine Quality Dataset

This lab demonstrates how to use [ML Metadata (MLMD)](https://www.tensorflow.org/tfx/guide/mlmd) to track artifacts, executions, and events across a data validation pipeline. It is based on [Lab1 from the MLOps MLMD Labs](https://github.com/raminmohammadi/MLOps/tree/main/Labs/MLMD_Labs/Lab1), with the following modifications.

## Modifications from the Original Lab

| Aspect | Original Lab | This Lab |
|---|---|---|
| **Dataset** | Chicago Taxi | UCI Wine Quality (Red) |
| **Storage backend** | Fake (in-memory) database | SQLite persistent database |
| **Execution types** | 1 (Data Validation) | 3 (StatisticsGen, SchemaGen, AnomalyDetection) |
| **Artifact types** | 2 (DataSet, Schema) + bonus Statistics | 4 (DataSet, Statistics, Schema, Anomalies) |
| **Anomaly detection** | Not included | Validates eval split against inferred schema |
| **Statistics visualization** | Not included | Visualizes train stats and train-vs-eval comparison |
| **Context properties** | Basic (note) | Extended (note, pipeline_version, dataset_source) |
| **Lineage tracing** | Schema → Dataset (1 hop) | Schema → Statistics → Dataset (2 hops) |

## Project Structure

```
MLMD Lab/
├── README.md
├── requirements.txt
├── .gitignore
├── schema.pbtxt                    # TFDV-inferred schema for the Wine Quality dataset
├── MLMD_Wine_Quality.ipynb         # Main notebook
└── data/
    ├── train/
    │   └── data.csv                # Training split (80%, 1279 rows)
    └── eval/
        └── data.csv                # Evaluation split (20%, 320 rows)
```

## Dataset

The [UCI Wine Quality dataset (Red Wine)](https://archive.ics.uci.edu/ml/datasets/wine+quality) contains 1,599 samples with 11 physicochemical input features and 1 output variable (quality score 0–10):

- Fixed acidity, volatile acidity, citric acid
- Residual sugar, chlorides
- Free/total sulfur dioxide
- Density, pH, sulphates, alcohol
- **Quality** (target, integer 3–8)

The dataset is split 80/20 into train and eval sets using `sklearn.model_selection.train_test_split` with `random_state=42`.

## Setup

```bash
pip install -r requirements.txt
```

Then open and run `MLMD_Wine_Quality.ipynb`.

## Pipeline Steps Tracked in MLMD

1. **StatisticsGen** — Generates descriptive statistics from the training CSV using TFDV
2. **SchemaGen** — Infers a data schema from the training statistics
3. **AnomalyDetection** — Validates the evaluation split against the inferred schema and records any anomalies

Each step is recorded with proper input/output events, and all artifacts and executions are grouped under an Experiment context.

## References

- [ML Metadata Documentation](https://www.tensorflow.org/tfx/guide/mlmd)
- [MetadataStore API](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore)
- [Original Lab (raminmohammadi/MLOps)](https://github.com/raminmohammadi/MLOps/tree/main/Labs/MLMD_Labs/Lab1)
- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
