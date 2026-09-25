# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
I built the model as a Random Forest Classifier from scikit-learn's RandomForestClassifier. The model uses 100 trees and a fixed random state of 32 so that every run gives the same results. This was created for a machine learning pipeline that attempts to decide whether a person's yearly income is greater than $50,000 using data from the U.S. Census.

## Intended Use

The model is intended to classify whether a person's yearly income is over or under $50,000.

## Training Data

The model was trained on the Census Income dataset.
- There are about 32,500 rows
- Contains fields such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours worked per week, and country. 
- 80% of the data was used for training

## Evaluation Data
20% of the data was using for testing. The set was processed using one-hot encoder and label binarizer that were fit to the training data to keep the features representation consistent.

## Metrics
The model was evaluated using precision, recall, and F1 score. I used scikit-learn functions "recall_score", “precision_score”,  and "fbeta_score". On the held-out test set the model achieved a precision of 0.7194, a recall of 0.6240 and an F1 score of 0.6683. The model also looked at performance on slices of the data for each value of every feature. I noticed that the model performance varied based on feature sizes. For example, "Without-pay" produced perfect scores due to small sample size. Features with larger samples had more varying scores.

## Ethical Considerations
I feel as though the "race", "sex", and "native country" features can carry bias. Additionally, the slice-based performance showed that models metrics varied across different features. The model does not seem to perform well for all groups. 

## Caveats and Recommendations
The dataset is from 1994, so the data is heavily outdated. If this was ever to be used for a real-world project, a newer census should be recorded.
