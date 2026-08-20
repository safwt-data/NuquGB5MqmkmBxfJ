# NuquGB5MqmkmBxfJ
The project focuses primarily on machine learning modeling and feature importance analysis to predict customer happiness in a customer service company.

The project will begin with Exploratory Data Analysis (EDA), including an assessment of missing values and and a correlation matrix. Feature engineering will not be included, as the dataset contains only six variables and there are limited opportunities to derive meaningful additional features.

Several machine learning models will be trained and evaluated to compare their predictive performance. The Random Forest model will then be explored further through RandomizedSearchCV, where a predefined number of hyperparameter combinations will be randomly selected and evaluated using cross-validation and the best-performing hyperparameters will then be used to build the optimized model.

Lastly, feature importance analysis will be conducted using the optimized Random Forest model to identify which variables contribute most and least to its predictions. Permutation feature importance will also be applied to both the training and test datasets. This will help validate whether the variables identified as important provide stable predictive signals rather than reflecting noise. The analysis will show how strongly the model depends on each variable when predicting customer happiness.



