# Fraud Detection with Heterogeneous GNN

This project contains a production-oriented notebook for fraud detection on the IEEE-CIS dataset using a heterogeneous graph neural network. It combines transaction, card, and device entities into a graph, applies chronological splitting to avoid leakage, and trains a temporal GNN with neighbor sampling, class-imbalance handling, and evaluation focused on fraud-detection metrics.

## What the notebook does

- Loads engineered transaction and identity data from Google Drive.
- Builds a heterogeneous graph with transaction, card, and device nodes.
- Adds static and temporal edges so the model can learn both structural and time-aware patterns.
- Splits data chronologically into train, validation, and test sets.
- Trains a GNN with sampling-based mini-batches to keep memory use manageable.
- Evaluates the model with ROC-AUC, PR-AUC, F1, precision, recall, and confusion matrices.
- Includes follow-up analysis for score distributions, hard negatives, calibration, and cascade-style refinement.

## Repository contents

- `Copy_of_Yet_another_copy_of_Fraud_Detection_After_Feature_Engineering_v3.ipynb`: main notebook.
- `Notebook_Summary.md`: screenshot-friendly notebook walkthrough.
- `screenshots/`: extracted or manually added output images.

## Results and reporting

The notebook produces training logs, graph diagnostics, final test metrics, and visualization outputs that are useful for GitHub documentation or a project portfolio. The screenshots folder can be linked from the Markdown summary to show the most important results inline.

## Notes

- The workflow is designed for Colab or a similar GPU-enabled notebook environment.
- The README intentionally gives a high-level overview instead of a cell-by-cell explanation.

