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
import numpy as np
from interpret.glassbox import LogisticRegression as InterpretLogisticRegression
from interpret import show


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
    estimator=LogisticRegression(max_iter=1000, random_state=8),
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
    estimator=LinearSVC(tol=1e-5, random_state=8),
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
    estimator=RandomForestClassifier(random_state=8),
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
    estimator=SVC(kernel="rbf", random_state=8),
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
    estimator=xgb.XGBClassifier(eval_metric="logloss", random_state=8),
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

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Macro-F1
ax1.bar(metrics_df["Model"], metrics_df["Macro-F1"], color="steelblue")
ax1.set_ylim(0.68, 0.80)
ax1.set_ylabel("Macro-F1")
ax1.set_title("Macro-F1 by model")
ax1.tick_params(axis="x", rotation=20)

# Recall
ax2.bar(metrics_df["Model"], metrics_df["Recall (Private Class)"], color="orange")
ax2.set_ylim(0.55, 0.80)
ax2.set_ylabel("Recall (Private class)")
ax2.set_title("Recall by model")
ax2.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Pick and test the best model (using test dataset)

# %%
# Prediction
best_model = model_random_forest
Y_pred = best_model.predict(X_test)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_test, Y_pred)

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ## Analysis 
#
# ### Error analysis:
#
# En se basant sur la matrice de confusion du meilleur modèle (Random Forest sur test) :
#
# [[1187, 163], [98, 352]]
#
# Le modèle produit 163 faux positifs (images publiques classifiées privées) et 98 faux négatifs (images privées non détectées). Les faux négatifs sont les erreurs les plus critiques dans un privacy-warning system : ils correspondent à des images privées que le système n'aurait pas signalées à l'utilisateur.
#
# ### Critical discussion: 
#
# Le meilleur modèle dépend de la métrique considérée. Dans notre contexte, les métriques essentielles sont le Macro-F1 et le Recall de la classe private.
# Le Macro-F1 mesure la capacité du modèle à classer correctement chaque classe même lorsque le dataset est déséquilibré (ce qui est le cas ici : environ 3 fois plus d'images publiques que privées). Le Recall de la classe private représente la capacité du modèle à détecter toutes les images privées. Cette métrique est particulièrement critique, car dans un privacy-warning system, un faux négatif signifie que le modèle n'a pas alerté l'utilisateur sur une image qui était en réalité privée.
#
# En se basant sur le Macro-F1, RBF SVM (0.780) et Random Forest (0.778) sont essentiellement à égalité. En se basant uniquement sur le Recall de la classe private, Logistic Regression et Linear SVM sont supérieurs (0.74). Nous retenons Random Forest comme meilleur modèle, car il est beaucoup plus interprétable que RBF SVM.
#
# Un résultat notable est que les modèles linéaires simples (Logistic Regression, Linear SVM) obtiennent des performances très proches de Random Forest. En revanche, XGBoost, pourtant plus complexe, ne surpasse pas Random Forest ce qui suggère que sa complexité supplémentaire n'apporte pas de gain sur notre type de données (textuelles).
#
# Il existe un trade-off entre performance et simplicité. Random Forest offre un meilleur Macro-F1, mais est difficile à interpréter. Logistic Regression, avec un Macro-F1 légèrement inférieur (0.77 vs 0.78), est entièrement interprétable via ses coefficients et beaucoup plus rapide à entraîner.

# %% [markdown]
# # Topic 1 — User tags vs deep tags:
#
# Question:
# Are human/social tags or automatically generated visual tags more useful for predicting privacy?
#
# Compare:
#
# - user_tags only
# - deep_tags only
# - user_deep_tags

# %% [markdown]
# ## Load training, tests and validation datasets (for user + deep + user_deep tags)

# %% [markdown]
# ### User and Deep tags

# %%
vectorizer_user_deep_tags = TfidfVectorizer(max_features=5000)

# Get features (X) and target (Y) for training dataset (user and deep tags)
X_train_user_deep_tags, Y_train_user_deep_tags = get_features_target_from_tags_file(
    TRAIN_USER_DEEP_TAGS_PATH, vectorizer_user_deep_tags, False
)

# Get features (X) and target (Y) for test dataset (user and deep tags)
X_test_user_deep_tags, Y_test_user_deep_tags = get_features_target_from_tags_file(
    TEST_USER_DEEP_TAGS_PATH, vectorizer_user_deep_tags, True
)

# Get features (X) and target (Y) for validation dataset (user and deep tags)
X_val_user_deep_tags, Y_val_user_deep_tags = get_features_target_from_tags_file(
    VAL_USER_DEEP_TAGS_PATH, vectorizer_user_deep_tags, True
)

# List that contains all the metrics for all the models
metrics = []

# %% [markdown]
# ### User tags

# %%
vectorizer_user_tags = TfidfVectorizer(max_features=5000)

# Get features (X) and target (Y) for training dataset (user tags)
X_train_user_tags, Y_train_user_tags = get_features_target_from_tags_file(
    TRAIN_USER_TAGS_PATH, vectorizer_user_tags, False
)

# Get features (X) and target (Y) for test dataset (user tags)
X_test_user_tags, Y_test_user_tags = get_features_target_from_tags_file(
    TEST_USER_TAGS_PATH, vectorizer_user_tags, True
)

# Get features (X) and target (Y) for validation dataset (user tags)
X_val_user_tags, Y_val_user_tags = get_features_target_from_tags_file(
    VAL_USER_TAGS_PATH, vectorizer_user_tags, True
)

# %% [markdown]
# ### Deep tags

# %%
vectorizer_deep_tags = TfidfVectorizer(max_features=5000)


# Get features (X) and target (Y) for training dataset (deep tags)
X_train_deep_tags, Y_train_deep_tags = get_features_target_from_tags_file(
    TRAIN_DEEP_TAGS_PATH, vectorizer_deep_tags, False
)

# Get features (X) and target (Y) for test dataset (deep tags)
X_test_deep_tags, Y_test_deep_tags = get_features_target_from_tags_file(
    TEST_DEEP_TAGS_PATH, vectorizer_deep_tags, True
)

# Get features (X) and target (Y) for validation dataset (deep tags)
X_val_deep_tags, Y_val_deep_tags = get_features_target_from_tags_file(
    VAL_DEEP_TAGS_PATH, vectorizer_deep_tags, True
)

# %% [markdown]
# ## Train and Tune models from user + deep + user_deep tags

# %% [markdown]
# ### User and Deep tags

# %%
# Train and Tune Logistic Regression model

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=8),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train_user_deep_tags, Y_train_user_deep_tags)

model_logistic_regression_user_deep_tags = gs.best_estimator_


# Get the best hyperparameters
best_params = gs.best_params_
print(f"Best params: {best_params}")
best_params["random_state"] = 8

# Get the features name
feature_names = vectorizer_user_deep_tags.get_feature_names_out()

# Fed the best hyperparameters into the InterpretML model
model_interpret_user_deep_tags = InterpretLogisticRegression(
    max_iter=1000,
    feature_names=feature_names,
    **best_params,
)

# Train the InterpretML model (required to generate the explanation)
model_interpret_user_deep_tags.fit(X_train_user_deep_tags, Y_train_user_deep_tags)

# %% [markdown]
# ### User tags

# %%
# Train and Tune Logistic Regression model

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=8),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train_user_tags, Y_train_user_tags)

model_logistic_regression_user_tags = gs.best_estimator_


# Get the best hyperparameters
best_params = gs.best_params_
print(f"Best params: {best_params}")
best_params["random_state"] = 8

# Get the features name
feature_names = vectorizer_user_tags.get_feature_names_out()

# Fed the best hyperparameters into the InterpretML model
model_interpret_user_tags = InterpretLogisticRegression(
    max_iter=1000,
    feature_names=feature_names,
    **best_params,
)

# Train the InterpretML model (required to generate the explanation)
model_interpret_user_tags.fit(X_train_user_tags, Y_train_user_tags)

# %% [markdown]
# ### Deep tags

# %%
# Train and Tune Logistic Regression model

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=8),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train_deep_tags, Y_train_deep_tags)

model_logistic_regression_deep_tags = gs.best_estimator_


# Get the best hyperparameters
best_params = gs.best_params_
print(f"Best params: {best_params}")
best_params["random_state"] = 8

# Get the features name
feature_names = vectorizer_deep_tags.get_feature_names_out()

# Fed the best hyperparameters into the InterpretML model
model_interpret_deep_tags = InterpretLogisticRegression(
    max_iter=1000,
    feature_names=feature_names,
    **best_params,
)

# Train the InterpretML model (required to generate the explanation)
model_interpret_deep_tags.fit(X_train_deep_tags, Y_train_deep_tags)

# %% [markdown]
# ## Compare the models (using validation dataset)

# %% [markdown]
# ### Results for user and deep tags

# %%
# Prediction
Y_pred = model_interpret_user_deep_tags.predict(X_val_user_deep_tags)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(
    Y_val_user_deep_tags, Y_pred
)

metrics.append(
    ["User and Deep tags", acc, f1_macro, prec_private, rec_private, conf_matrix]
)

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Results for user tags

# %%
# Prediction
Y_pred = model_interpret_user_tags.predict(X_val_user_tags)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(
    Y_val_user_tags, Y_pred
)

metrics.append(["User tags", acc, f1_macro, prec_private, rec_private, conf_matrix])

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Results for deep tags

# %%
# Prediction
Y_pred = model_interpret_deep_tags.predict(X_val_deep_tags)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(
    Y_val_deep_tags, Y_pred
)

metrics.append(["Deep tags", acc, f1_macro, prec_private, rec_private, conf_matrix])

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ### Summary

# %%
metrics_df = pd.DataFrame(
    metrics,
    columns=[
        "Tags",
        "Accuracy",
        "Macro-F1",
        "Precision (Private class)",
        "Recall (Private Class)",
        "Confusion matrix",
    ],
)
fig, axs = plt.subplots(metrics_df.shape[0], figsize=(20, 20))


for index, row in metrics_df.iterrows():
    disp = ConfusionMatrixDisplay(
        confusion_matrix=row["Confusion matrix"], display_labels=["Public", "Private"]
    )
    disp.plot(ax=axs[index], cmap=plt.cm.Blues)
    axs[index].set_title("Confusion Matrix for " + row["Tags"])


plt.show()

metrics_df

# %% [markdown]
# ## Pick and test the best model (using test dataset)

# %%
# Prediction
best_model = model_interpret_user_deep_tags
Y_pred = best_model.predict(X_test_user_deep_tags)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(
    Y_test_user_deep_tags, Y_pred
)

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ## Analysis of Key Features

# %% [markdown]
# ### User tags

# %%
explanation = model_interpret_user_tags.explain_global(name="User tags")
show(explanation)

# %% [markdown]
# ### Deep tags

# %%
explanation = model_interpret_deep_tags.explain_global(name="Deep tags")
show(explanation)

# %% [markdown]
# ### User and Deep tags

# %%
explanation = model_interpret_user_deep_tags.explain_global(name="User Deep tags")
show(explanation)

# %% [markdown]
# ## Analysis
#
# ### Error analysis:
#
# On se base sur les résultats obtenus avec le validation dataset.
#
# Les deep_tags produisent le plus de faux positifs (280) mais relativement peu de faux négatifs (129). Ainsi, le modèle préfère sur-alerter plutôt que manquer des images privées. Cela explique son recall élevé (0.72) mais sa précision faible (0.55). À l'inverse, user_tags est moins prudent. Avec ce modèle, on a moins de fausses alertes (188) mais plus d'images privées manquées (149). La combinaison user_deep_tags offre le meilleur compromis global.
#
# Les faux négatifs dans les trois cas correspondent probablement aux mêmes images : des photos privées sans contenu sexuel (intérieurs de maison, documents...) que le modèle ne reconnaît pas comme privé, car ces patterns sont sous-représentés voire absent dans les features dominantes.
#
# ### Critical discussion:
#
# Nous avons utilisé Logistic Regression plutôt que Random Forest, car ce modèle donne de bons résultats tout en étant entièrement interprétable via ses coefficients.
#
# En se basant sur le Macro-F1, la hiérarchie est user_deep_tags et user_tags un peu près au même niveau, suivi de deep_tags. La différence entre user_deep_tags et user_tags est négligeable. Cependant, combiner les deux sources à un intérêt visible sur le Recall (0.74 pour user_deep_tags vs 0.68 pour user_tags). Ainsi, les deep_tags augmentent la sensibilité du modèle, mais génèrent aussi plus de faux positifs.
#
# Les graphiques précédents révèlent que le modèle apprend deux types de features. Du côté "private", on a naked, nude, sexy, maillot, trunks qui sont tous liés à du contenu sexuel. Du côté "public", on a graffiti, sign, church, fence qui sont des espaces urbains extérieurs. Le modèle n'apprend donc pas la privacy au sens large, mais distingue "contenu sexuel" de "scène publique générique".
#
# Ce résultat s'explique par le contenu du dataset : les images "private" dans PrivacyAlert correspondent quasi-exclusivement à du contenu sexuel et à du contenu LGBTQ+, tandis que les images "public" sont des scènes génériques extérieures. Ce n'est pas un échec de généralisation du modèle. Le problème principal est que le dataset représente une définition très étroite et biaisée de la privacy.
#
# Une conséquence de cela est que le tag gay figure parmi les features les plus discriminantes pour la classe "private" dans le modèle user_deep_tags. Un tel modèle risque d'associer systématiquement le contenu LGBTQ+ à du contenu privé, ce qui est très problématique.

# %% [markdown]
# # Topic 6 — Decision threshold and social cost
#
# Question:
# In a privacy-warning system, should we prefer more warnings or fewer missed private images?
#
# Many classifiers produce probabilities or scores.
# The default threshold is often:
#
# - 0.5
#
# But you can compare:
#
# - 0.2
# - 0.3
# - 0.5
# - 0.7
# - 0.8

# %% [markdown]
# ## Train and tune best resulting machine learning algorithm (from Topic 3)

# %%
# Train and Tune Random Forest model

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced"],
}

gs = GridSearchCV(
    estimator=RandomForestClassifier(random_state=8),
    param_grid=param_grid,
    scoring="recall",
    n_jobs=-1,
)
gs.fit(X_train, Y_train)

model_random_forest = gs.best_estimator_

# %% [markdown]
# ## Results on Validation dataset

# %%

y_proba = model_random_forest.predict_proba(X_val)[:, 1]

thresholds = [0.2, 0.3, 0.5, 0.7, 0.8]
metrics = []


for t in thresholds:
    y_pred = (y_proba >= t).astype(int)

    acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_val, y_pred)

    metrics.append([t, acc, f1_macro, prec_private, rec_private, conf_matrix])

    #print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ## Compare the models (using validation dataset)

# %%
metrics_df = pd.DataFrame(
    metrics,
    columns=[
        "Threshold",
        "Accuracy",
        "Macro-F1",
        "Precision (Private class)",
        "Recall (Private Class)",
        "Confusion matrix",
    ],
)
fig, axs = plt.subplots(metrics_df.shape[0], figsize=(25, 25))


for index, row in metrics_df.iterrows():
    disp = ConfusionMatrixDisplay(
        confusion_matrix=row["Confusion matrix"], display_labels=["Public", "Private"]
    )
    disp.plot(ax=axs[index], cmap=plt.cm.Blues)
    axs[index].set_title("Confusion Matrix for " + str(row["Threshold"]))


plt.show()

metrics_df

# %% [markdown]
# ## Analyse
#
#
# ### Differences for Precision and Recall
#
# Ce que nous pouvons remarquer :
#
# La Précision augmente avec le threshold : elle passe de 44 % à 0.2 jusqu'à 86 % à 0.8. En montant le seuil, le modèle ne déclenche une alerte que lorsqu'il est très confiant => il produit moins de fausses alertes
# Le Recall diminue inversement : il chute de 88 % à 0.2 à 36 % à 0.8. Un threshold élevé laisse passer beaucoup d'images privées sans alertes
#
# Il existe donc un échange entre nombres de faux positifs (threshold bas) et nombre de faux négatifs (threshold haut).
#
# Considérant le type de données que l'application traite, il semble très largement préférable de favoriser le Recall par rapport à la Précision. De manière générale, on préfère largement recevoir une alerte pour rien plutôt que de laisser une image privée être publiée.
# Néanmoins, il est important de noter que recevoir trop d'alertes a la publication peut entraîner un épuisement de la part de l'utilisateur qui soit retire l'application soit appuie sur "quand même publier" sans regarder la raison de l'alerte.
#
# ### Best threshold choice
#
# En prenant tout cela en compte, nous avons choisi un threshold de **0.3**, malgré la probabilité de recevoir une fausse alerte de 25 %, la probabilité de laisser passer une image privée en public est a seulement 16 %, c'est-à-dire 2 fois moins qu'avec une threshold de 0,5.

# %% [markdown]
# ## Test sur le Test dataset

# %%
y_proba = model_random_forest.predict_proba(X_test)[:, 1]

y_pred = (y_proba >= 0.3).astype(int)

acc, f1_macro, prec_private, rec_private, conf_matrix = compute_metrics(Y_test, y_pred)

print_metrics(acc, f1_macro, prec_private, rec_private, conf_matrix)

# %% [markdown]
# ## Conclusion
#
# Évalué sur le dataset de test, le modèle avec un threshold de 0.3 atteint un Recall de 88% : 8 images privées sur 10 déclenchent correctement une alerte avant publication. Les 54 faux négatifs restants représentent 12% des images privées.
# En contrepartie, la Précision de 51% implique 381 fausses alertes, soit environ 1 publication sur 4 faussement signalée. Ce niveau est acceptable dans ce contexte, car chaque fausse alerte reste gérable par l'utilisateur, contrairement à une image privée publiée silencieusement.
# Le Macro-F1 de 73% et l'accuracy de 76% confirment des performances globalement solides, malgré le déséquilibre volontaire entre Recall et Precision.
# Ce threshold constitue ainsi le point d'équilibre optimal pour une application de protection de la vie privée : il minimise le risque irréversible (laisser passer une image privée) tout en maintenant un taux de fausses alertes raisonnable pour ne pas éroder la confiance et l'usage de l'application.

# %% [markdown]
#
