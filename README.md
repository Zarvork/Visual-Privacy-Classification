# Visual Privacy Classification

Can we predict whether an image is **public** or **private** from precomputed features alone? This project trains and compares classical machine learning models on the [PrivacyAlert](https://doi.org/10.1609/icwsm.v16i1.19387) dataset (tags, scene features and object detections), and studies what the models actually learn rather than only their scores.

## Questions studied

1. **Model comparison:** Logistic Regression, Linear SVM, RBF SVM, Random Forest, XGBoost and k-NN.
2. **User tags vs deep tags:** which is more useful for predicting privacy?
3. **Tags vs scenes vs objects:** which feature family, or combination, works best?
4. **Decision threshold:** how should the threshold be set in a warning system where missing a private image is costly?

Metrics: accuracy, macro-F1, precision and recall of the `private` class, and confusion matrix. Recall on `private` matters most, since a false negative means a private image goes unflagged.

## Approach

- **Tags:** user and/or deep tags, vectorized with TF-IDF (5000 features)
- **Scenes:** scene tags with confidence > 0.99, TF-IDF (300 features)
- **Objects:** counts of the 80 detected object categories (confidence > 0.8), log-scaled
- Models tuned with `GridSearchCV` on the training split, compared on the validation split, and the chosen model evaluated on the test split
- Interpretation with an [InterpretML](https://interpret.ml) Logistic Regression to inspect the most discriminative tags

## Installation

This project uses [uv](https://docs.astral.sh/uv/). The data is included in the repository.

```bash
git clone https://github.com/Zarvork/Visual-Privacy-Classification.git
cd Visual-Privacy-Classification
uv sync
```

## Usage

```bash
uv run jupyter lab VPC.ipynb
```

`VPC.py` is the same notebook in [jupytext](https://jupytext.readthedocs.io) percent format.

## Authors

- Anis Feore
- Roman Miralves
- Johan Emmanuelli
