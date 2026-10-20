# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This machine learning model is a Random Forest Classifier trained to predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes. The model was trained using scikit-learn version 1.5.1 with 100 decision trees (`n_estimators=100`) and a fixed random seed of 42. Categorical features are preprocessed using a fitted OneHotEncoder, and continuous numerical features are passed directly without scaling.

## Intended Use
The primary intended use of this model is educational and research purposes to demonstrate end-to-end machine learning pipelines and API deployment using FastAPI. It is intended to classify individuals into binary income brackets (`<=50K` or `>50K`). It is not intended for high-stakes decision-making such as credit scoring, loan approvals, hiring screening, or insurance underwriting.

## Training Data
The model was trained on an 80% split of the publicly available Census Bureau Income dataset (`census.csv`), consisting of 26,048 records. Features include demographic and socioeconomic indicators such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours worked per week, and country of origin. Target labels were binarized into 0 (`<=50K`) and 1 (`>50K`).

## Evaluation Data
Model evaluation was performed on a held-out test split comprising the remaining 20% of the dataset (6,513 records), created using a random 80-20 train-test split (`test_size=0.20`, `random_state=42`). The test split was held separate throughout feature encoding and model training to evaluate performance on unseen data.

## Metrics
The model was evaluated using binary classification metrics focused on positive income prediction (`>50K`):
- **Precision**: 0.7419 — Of all individuals predicted to earn more than $50,000, approximately 74.2% actually do.
- **Recall**: 0.6384 — The model identifies approximately 63.8% of all true high-earning individuals in the evaluation dataset.
- **F1 Score**: 0.6863 — The harmonic mean balancing precision and recall.

Slicing evaluations were generated across distinct categorical groups (recorded in `slice_output.txt`). The evaluation showed consistent predictions across common occupational and educational segments, while displaying variance in recall on sparsely represented demographic slices.

## Ethical Considerations
The dataset originates from historical U.S. Census survey data and reflects socioeconomic patterns and disparities present at the time of collection. Attributes such as `race`, `sex`, and `native-country` are utilized by the model, which introduces the risk of perpetuating historical wage gaps or demographic biases if applied in unmonitored settings. Performance slicing was conducted to monitor potential disparities across protected categories.

## Caveats and Recommendations
Because the source dataset was collected in 1994, the nominal $50,000 income threshold does not account for modern economic inflation, shifts in currency purchasing power, or changes in employment demographics. Future iterations should incorporate inflation adjustments, calibrate probability thresholds for higher recall, and explore fairness-constrained algorithms to mitigate demographic bias.