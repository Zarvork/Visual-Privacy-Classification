# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ml (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Vocabulary
#
# - Accuracy : The overall percentage of correct answers (not very important here).
# - Macro-F1 : The unweighted average of the F1 scores for each class (very important).
# - Precision for the private class : Of all the images I flagged as Private, how many were actually private? (Avoid false positives) (mid important).
# - Recall for the private class : Of all the private images, how many have been labeled as private? (Avoid false negative) (very very important).
# - Confusion matrix : The confusion matrix is used to evaluate the performance of a classification model by comparing the predicted values with the actual values in a dataset (very very important). 

# %% [markdown]
# # Import all necessary libraries

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# %% [markdown]
# # Define the path of useful files

# %%
# File that assigns a class to each image (public = 0 or private = 1)
LABELS_PATH = "data/curated_privacyalert/annotations/labels.csv"

# File that contains image for testing
TEST_WITH_LABELS_2CLASSES_PATH = (
    "data/curated_privacyalert/annotations/splits/test_with_labels_2classes.csv"
)
TEST_SPLIT_1_LABELS_ONLY = (
    "data/curated_privacyalert/annotations/splits/test_split_1_labels_only.csv"
)

# File that contains image for training
TRAIN_WITH_LABELS_2CLASSES_PATH = (
    "data/curated_privacyalert/annotations/splits/train_with_labels_2classes.csv"
)
TRAIN_SPLIT_1_LABELS_ONLY = (
    "data/curated_privacyalert/annotations/splits/train_split_1_labels_only.csv"
)

# File that contains image for validation
VAL_WITH_LABELS_2CLASSES_PATH = (
    "data/curated_privacyalert/annotations/splits/val_with_labels_2classes.csv"
)
VAL_SPLIT_1_LABELS_ONLY = (
    "data/curated_privacyalert/annotations/splits/val_split_1_labels_only.csv"
)

# File that contains all deep tags
ALL_DEEP_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/deep_tags/dt_plus_ut_overall1_2.tsv"
)

# File that contains deep tags of testing images
TEST_DEEP_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/deep_tags/dt_plus_ut_test1_2.tsv"
)

# File that contains deep tags of training images
TRAIN_DEEP_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/deep_tags/dt_plus_ut_train1_2.tsv"
)

# File that contains deep tags of validation images
VAL_DEEP_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/deep_tags/dt_plus_ut_val1_2.tsv"
)

# File that contains all user and deep tags
ALL_USER_DEEP_TAGS_PATH = "data/curated_privacyalert/annotations/tags/user_deep_tags/dt_plus_ut_overall1_2.tsv"

# File that contains user and deep tags of testing images
TEST_USER_DEEP_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/user_deep_tags/dt_plus_ut_test1_2.tsv"
)

# File that contains user and deep tags of training images
TRAIN_USER_DEEP_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/user_deep_tags/dt_plus_ut_train1_2.tsv"
)

# File that contains user and deep tags of validation images
VAL_USER_DEEP_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/user_deep_tags/dt_plus_ut_val1_2.tsv"
)

# File that contains all user tags
ALL_USER_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/user_tags/dt_plus_ut_overall1_2.tsv"
)

# File that contains user tags of testing images
TEST_USER_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/user_tags/dt_plus_ut_test1_2.tsv"
)

# File that contains user tags of training images
TRAIN_USER_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/user_tags/dt_plus_ut_train1_2.tsv"
)

# File that contains user tags of validation images
VAL_USER_TAGS_PATH = (
    "data/curated_privacyalert/annotations/tags/user_tags/dt_plus_ut_val1_2.tsv"
)


# %% [markdown]
# # Helper functions

# %%
def get_features_target_from_tags_file(path, vectorizer, is_fitted=False):
    # Load data
    data_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["idx", "label", "image_id", "tags"],
    )
    # Preparing features (X) and target (Y)
    X_raw = data_df["tags"].fillna("")
    Y = data_df["label"]
    # Transform text to numbers
    if not is_fitted:
        X = vectorizer.fit_transform(X_raw)
    else:
        X = vectorizer.transform(X_raw)

    return X, Y


# %%
def compute_metrics(Y_test, Y_pred):
    # Compute several metrics
    acc = accuracy_score(Y_test, Y_pred)
    f1_macro = f1_score(Y_test, Y_pred, average="macro")
    prec_private = precision_score(Y_test, Y_pred, pos_label=1)
    rec_private = recall_score(Y_test, Y_pred, pos_label=1)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    return acc, f1_macro, prec_private, rec_private, conf_matrix


# %%
def print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix):
    # Print result metrics
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1_macro:.4f}")
    print(f"Precision (Private class) : {prec_private:.4f}")
    print(f"Recall (Private Class) : {rec_private:.4f}")
    # Print confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=["Public", "Private"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


# %% [markdown]
# # Topic 3 — Comparison of classical models
#
# Question:
#
# Which classical machine learning model works best?
#
# Compare several models, for example:
#
# - Logistic Regression
# - Linear SVM
# - RBF SVM
# - Random Forest
# - XGBoost
# - k-NN

# %% [markdown]
# ## Load training, tests and validation datasets (user and deep tags)

# %%
vectorizer = TfidfVectorizer(max_features=5000)
# Get features (X) and target (Y) for training dataset (user and deep tags)
X_train, Y_train = get_features_target_from_tags_file(
    TRAIN_USER_DEEP_TAGS_PATH, vectorizer, False
)

# Get features (X) and target (Y) for test dataset (user and deep tags)
X_test, Y_test = get_features_target_from_tags_file(
    TEST_USER_DEEP_TAGS_PATH, vectorizer, True
)

# Get features (X) and target (Y) for validation dataset (user and deep tags)
X_val, Y_val = get_features_target_from_tags_file(
    VAL_USER_DEEP_TAGS_PATH, vectorizer, True
)

# List that contains all the metrics for all the models
metrics = []

# %% [markdown]
# ## Train and Tune different machine learning models

# %% [markdown]
# ### Train Logistic Regression model

# %%
# Train and Tune Logistic Regression model

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train, Y_train)

model_logistic_regression = gs.best_estimator_

# %% [markdown]
# ### Train Linear SVM model

# %%
# Train and Tune Linear SVM model

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=LinearSVC(tol=1e-5),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train, Y_train)

model_linear_svm = gs.best_estimator_

# %% [markdown]
# ### Train Random Forest model

# %%
# Train and Tune Random Forest model

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train, Y_train)

model_random_forest = gs.best_estimator_

# %% [markdown]
# ### Train k-NN model

# %%
# Train and Tune k-NN model

param_grid = {
    "n_neighbors": [3, 5, 7, 11],
    "weights": ["uniform", "distance"],
}

gs = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train, Y_train)

model_knn = gs.best_estimator_

# %% [markdown]
# ### Train RBF SVM model

# %%
# Train and Tune RBF SVM model

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train, Y_train)

model_rbf_svm = gs.best_estimator_

# %% [markdown]
# ### Train XGBoost model

# %%
# Train and Tune XGBoost model

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 6],
    "learning_rate": [0.01, 0.1],
    "scale_pos_weight": [1, 3],
}

gs = GridSearchCV(
    estimator=xgb.XGBClassifier(eval_metric="logloss"),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train, Y_train)

model_xgboost = gs.best_estimator_

# %% [markdown]
# ## Compare the models (using validation dataset)

# %% [markdown]
# ### Results for Logistic Regression model

# %%
# Prediction
Y_pred = model_logistic_regression.predict(X_val)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_val, Y_pred)

metrics.append(
    ["Logistic Regression", acc, f1_macro, prec_private, rec_private, conf_matrix]
)

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Results for Linear SVM model

# %%
# Prediction
Y_pred = model_linear_svm.predict(X_val)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_val, Y_pred)

metrics.append(["Linear SVM", acc, f1_macro, prec_private, rec_private, conf_matrix])

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Results for Random Forest model

# %%
# Prediction
Y_pred = model_random_forest.predict(X_val)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_val, Y_pred)

metrics.append(["Random Forest", acc, f1_macro, prec_private, rec_private, conf_matrix])

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Results for k-NN model

# %%
# Prediction
Y_pred = model_knn.predict(X_val)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_val, Y_pred)

metrics.append(["k-NN", acc, f1_macro, prec_private, rec_private, conf_matrix])

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Results for RBF SVM

# %%
# Prediction
Y_pred = model_rbf_svm.predict(X_val)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_val, Y_pred)

metrics.append(["RBF SVM", acc, f1_macro, prec_private, rec_private, conf_matrix])

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Results for XGBoost

# %%
# Prediction
Y_pred = model_xgboost.predict(X_val)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_val, Y_pred)

metrics.append(["XGBoost", acc, f1_macro, prec_private, rec_private, conf_matrix])

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Summary

# %%
metrics_df = pd.DataFrame(
    metrics,
    columns=[
        "Model",
        "Accuracy",
        "Macro-F1",
        "Precision (Private class)",
        "Recall (Private Class)",
        "Confusion matrix",
    ],
)
fig, axs = plt.subplots(metrics_df.shape[0], figsize=(30, 30))


for index, row in metrics_df.iterrows():
    disp = ConfusionMatrixDisplay(
        confusion_matrix=row["Confusion matrix"], display_labels=["Public", "Private"]
    )
    disp.plot(ax=axs[index], cmap=plt.cm.Blues)
    axs[index].set_title("Confusion Matrix for " + row["Model"])


plt.show()

metrics_df

# %% [markdown]
# ## Pick and test the best model (using test dataset)

# %%
# Prediction
best_model = model_random_forest
Y_pred = best_model.predict(X_test)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_test, Y_pred)

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)
