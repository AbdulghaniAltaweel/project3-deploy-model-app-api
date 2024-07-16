# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a random forest classifier from scikit-learn used for classification. Hyperparameters are the default of the classifier.

## Intended Use
Model is used to classify the salary of employees based on some informations. 

## Training Data
Census Bureau dataset is used for training the model https://archive.ics.uci.edu/dataset/20/census+income


## Evaluation Data
Census Bureau dataset is used for also for evaluation of the model https://archive.ics.uci.edu/dataset/20/census+income. The dataset is splitted in 8:2 ratio for training:evaluation respectively. 

## Metrics
The following metrics were used to evaluate the model performance: 
- Precision: 0.75
- Recall: 0.61
- F1_score: 0.67

## Ethical Considerations
The code in this project is mainly copied from https://github.com/udacity/nd0821-c3-starter-code.git and released under their license https://github.com/udacity/nd0821-c3-starter-code?tab=License-1-ov-file#readme

## Caveats and Recommendations
Consider that the model is trained on Census Bureau dataset which is collected in the USA. This mean that the model could not predict well for data collected in another country. 