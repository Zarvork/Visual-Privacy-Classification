# Evaluation EPITA Image 2 — Visual Privacy Classification

This repository contains the material for the EPITA Image course evaluation project.

## Topic

The project focuses on **visual privacy prediction**.

The practical question is:

> Can we predict whether an image is public or private using precomputed features?

This is a realistic problem: before someone shares an image online, a system could warn them if the image may contain private or sensitive visual content.

Examples of potentially private visual cues include:

- people in an indoor setting;
- bedrooms or bathrooms;
- computer screens;
- documents;
- phones;
- cars or license plates;
- home interiors;
- private social events.

The goal is not only to obtain a good classification score, but also to understand **what the model learns**.

---

## Scientific motivation

Visual privacy is difficult because an image is not private only because of one object.

For example:

- `person + street` may be public;
- `person + bedroom` may be private;
- `computer` alone is ambiguous;
- `computer + document + indoor scene` may be private;
- `home interior` may be private;
- `group + beach` may be public or private depending on context.

The central scientific question is therefore:

> Is visual privacy predictable from visual cues alone, or does it depend too much on social context?

This project asks students to investigate this question using classical machine learning methods.

---

## Dataset

The main dataset is **PrivacyAlert**, introduced by Zhao et al. (2022).

PrivacyAlert is a dataset for **image privacy prediction**. It was built from Flickr images and annotated with binary privacy labels:

```text
public
private
```

The dataset was designed to support research on whether image content and image metadata can be used to predict privacy sensitivity.

For this project, we use a curated version of PrivacyAlert distributed with the artifact of Xompero and Cavallaro (2025), *Learning Privacy from Visual Entities*.

Students do **not** work directly on the original images.  
They work on already extracted information:

- labels and train/validation/test splits;
- user tags;
- deep tags;
- combined user + deep tags;
- scene features;
- object features.

---

## Files provided in this repository

The relevant files are:

```text
data/
├── curated_imageprivacy_datasets.zip
├── objects_privacyalert.zip
└── scenes_privacyalert.zip
```

For this project, use only the **PrivacyAlert** part of the data from the Zenodo repository of their paper.

Do not use:

- PicAlert;
- VISPR;
- IPD;
- graph data;
- original images.

---

## What the files contain

### `curated_imageprivacy_datasets.zip`

This archive contains the curated datasets, including PrivacyAlert.

For this project, the useful folder is:

```text
curated_privacyalert/
```

The most useful files are:

```text
curated_privacyalert/
├── annotations/
│   ├── labels.csv
│   └── labels_splits.csv
├── splits/
│   ├── train_with_labels_2classes.csv
│   ├── val_with_labels_2classes.csv
│   └── test_with_labels_2classes.csv
└── tags/
    ├── user_tags/
    ├── deep_tags/
    └── user_deep_tags/
```

The target variable is binary:

```text
public
private
```

---

## Where the features come from

The students do **not** compute these features. They are already provided.

The general pipeline is:

```text
original image
→ previous models / metadata extraction
→ precomputed tags, scene features, and object features
→ classical ML model trained by students
→ public/private prediction
```

### Labels

The `public/private` labels come from the PrivacyAlert dataset.

They are the ground truth labels that students try to predict.

### User tags

`user_tags` are tags or metadata associated with the image by users.

They may capture social or contextual information, for example:

```text
family
friends
travel
party
home
wedding
vacation
```

These tags are not purely visual: they may reflect how the image was described or shared.

### Deep tags

`deep_tags` are tags automatically generated from the image by a computer vision system.

They may correspond to objects, scenes, or visual concepts detected in the image, for example:

```text
person
bedroom
car
screen
phone
dog
street
food
```

In the original PrivacyAlert work, tags are modeled with **BERT** when used as textual input.

### Scene features

`scenes_privacyalert.zip` contains precomputed scene-related features.

These features describe the probable type of environment in the image, for example:

```text
bedroom
bathroom
office
street
beach
restaurant
classroom
kitchen
living room
```

In the PrivacyAlert work, scene information is modeled with a **ResNet-50** scene model.  
In the later *Learning Privacy from Visual Entities* artifact, scene tags/features are described as coming from a CNN trained on **Places365**.

### Object features

`objects_privacyalert.zip` contains precomputed object-related features.

These features describe objects detected in the image, for example:

```text
person
bed
laptop
cell phone
book
car
toilet
backpack
screen
chair
```

In the PrivacyAlert work, object information is modeled with **ResNet-101**.  
In the later *Learning Privacy from Visual Entities* artifact, object tags/features are described as coming from a CNN trained on **ImageNet**.

### Combined tags

`user_deep_tags` combine user tags and deep tags.

They can be interpreted as a richer representation that mixes:

```text
social/contextual metadata + automatically detected visual concepts
```

---

## Why tags, scenes, and objects matter

These feature families allow us to ask several questions:

> Is privacy better predicted by user-provided context or by automatically detected visual content?

> Are scene features enough to detect private images?

> Are object features enough?

> Does combining tags, scenes, and objects improve performance?

For example, an image tagged as:

```text
family, birthday, home
```

may be considered private even if the visual objects are not obviously sensitive.

An image with automatically detected concepts such as:

```text
bedroom, person, laptop
```

may also be considered private, even without user-provided tags.

Students can therefore compare:

```text
user_tags only
deep_tags only
user_deep_tags
scenes only
objects only
combined features
```

---

## Important restriction

For this project, students should **not**:

- download the original images;
- train a deep neural network;
- fine-tune a CNN, ViT, CLIP, DINO, or any foundation model;
- compute new embeddings;
- use graph neural networks;
- use the other datasets unless explicitly instructed.

The project is about **classical machine learning on precomputed features**.

---

## Allowed methods

Students may use methods seen in class, including:

- Logistic Regression;
- SVM;
- k-NN;
- Random Forest;
- XGBoost;
- PCA;
- t-SNE;
- UMAP;
- k-means;
- hierarchical clustering;
- Gaussian Mixture Models;
- HDBSCAN.

---

## Main task

Given precomputed features, predict whether an image is:

```text
public
```

or:

```text
private
```

This is a binary classification task.

---

## Required metrics

Students should report at least:

- accuracy;
- macro-F1;
- precision for the `private` class;
- recall for the `private` class;
- confusion matrix.

The `private` class is especially important.

In a privacy-warning system, a false negative means:

> the model failed to warn the user about an image that was actually private.

Therefore, the **recall of the private class** is a central metric.

---

## Suggested project topics

Each group should focus on at least two of the following questions.

### Topic 1 — User tags vs deep tags

Question:

> Are human/social tags or automatically generated visual tags more useful for predicting privacy?

Compare:

```text
user_tags only
deep_tags only
user_deep_tags
```

Expected discussion:

- Do user tags capture useful social context?
- Do deep tags capture useful visual cues?
- Does combining both improve performance?
- Are some tags too directly correlated with the label?

---

### Topic 2 — Tags vs scenes vs objects

Question:

> Which feature family is most useful for predicting visual privacy?

Compare:

```text
tags
scene features
object features
tags + scenes
tags + objects
tags + scenes + objects
```

Expected discussion:

- Are scenes enough?
- Are objects enough?
- Do user/deep tags add contextual information?
- Does combining everything improve performance or overfit?

---

### Topic 3 — Comparison of classical models

Question:

> Which classical machine learning model works best?

Compare several models, for example:

```text
Logistic Regression
Linear SVM
RBF SVM
Random Forest
XGBoost
k-NN
```

Expected discussion:

- Is a linear model already strong?
- Does XGBoost improve performance?
- Is the best model also the most interpretable?
- Is there a trade-off between performance and simplicity?

---

### Topic 4 — Interpretability of tags, scenes, and objects

Question:

> Which features are most associated with private or public images?

Use interpretable models such as:

```text
Logistic Regression
Linear SVM
Random Forest
XGBoost
```

Expected outputs:

- top tags associated with `private`;
- top tags associated with `public`;
- top scenes associated with `private`;
- top objects associated with `private`;
- discussion of ambiguous features.

Example:

```text
bedroom  → probably private
street   → probably public
person   → ambiguous
laptop   → context-dependent
family   → possibly private
travel   → ambiguous
```

Critical question:

> Does the model really learn privacy, or does it learn shortcuts such as “indoor = private” or “family = private”?

---

### Topic 5 — PCA, UMAP, and dimensionality reduction

Question:

> Can we reduce the feature space without losing predictive performance?

Possible experiments:

```text
raw features
PCA 5
PCA 10
PCA 25
PCA 50
PCA 100
```

Expected outputs:

- performance as a function of the number of PCA components;
- 2D visualization using PCA, UMAP, or t-SNE;
- discussion of whether `public` and `private` images are separable in the feature space.

Critical question:

> Is privacy captured by a few global dimensions, or by many small and specific visual cues?

---

### Topic 6 — Decision threshold and social cost

Question:

> In a privacy-warning system, should we prefer more warnings or fewer missed private images?

Many classifiers produce probabilities or scores.  
The default threshold is often:

```text
0.5
```

But you can compare:

```text
0.2
0.3
0.5
0.7
0.8
```

Expected discussion:

- What happens to precision when the threshold changes?
- What happens to recall for the private class?
- Which threshold would be appropriate for an assistant that warns users before publication?

Critical question:

> Is it worse to produce too many false alerts, or to miss genuinely private images?

---

### Topic 7 — Error analysis and ambiguous cases

Question:

> What do the model errors reveal about visual privacy?

Analyze:

```text
false positives
false negatives
```

A false positive is:

```text
label = public
prediction = private
```

A false negative is:

```text
label = private
prediction = public
```

Expected analysis:

- Which tags appear often in false positives?
- Which tags appear often in false negatives?
- Which scene features appear often in errors?
- Which object features appear often in errors?
- Are some errors understandable?
- Are some labels themselves ambiguous?

Critical question:

> Are the model’s errors only technical mistakes, or do they reveal that privacy is socially contextual?

---

## Expected output

Each group should submit:

1. a reproducible notebook;
2. a table of scores;
3. a confusion matrix;
4. at least one visualization;
5. an error analysis;
6. a short critical discussion.

The final answer should not only say:

> Our model gets the best score.

It should also answer:

> What does the model learn, and where does it fail?

---

## Suggested evaluation grid

| Criterion | Points |
|---|---:|
| Reproducible pipeline | 4 |
| Correct use of metrics | 4 |
| Methodological comparison | 4 |
| Visualization | 3 |
| Error analysis | 3 |
| Critical discussion | 2 |
| **Total** | **20** |

---

## Minimal workflow

A typical workflow may look like this:

```text
load labels and splits
load tags, scenes, and/or object features
merge features with labels
train model on train split
tune on validation split
evaluate on test split
compute metrics
analyze errors
interpret results
```

---

## Practical interpretation

This project simulates a simple privacy assistant.

Such a system could be used to warn users before sharing images online:

> This image may contain private visual content. Please check before posting.

However, the project also asks whether such a system is reliable.

The key limitation is that privacy is not only visual.  
It also depends on:

- who is sharing the image;
- who appears in the image;
- whether people gave consent;
- the publication context;
- the intended audience;
- cultural and social norms.

---

## References

Zhao, C., Mangat, J., Koujalgi, S., Squicciarini, A. C., & Caragea, C. (2022). PrivacyAlert: A dataset for image privacy prediction. *Proceedings of the International AAAI Conference on Web and Social Media, 16*(1), 1352–1361. https://doi.org/10.1609/icwsm.v16i1.19387

Xompero, A., & Cavallaro, A. (2025). Learning privacy from visual entities. *Proceedings on Privacy Enhancing Technologies, 2025*(3), 261–281. https://doi.org/10.56553/popets-2025-0098
