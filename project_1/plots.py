import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from project_1.dataset import load_data
from project_1.modeling.train import split_data
from project_1.modeling.train import train_model
from project_1.modeling.predict import make_predictions



def correlation_matrix(df):
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr))
    correlation_matrix = sns.heatmap(corr, center=0,
            mask=mask,
              linewidths=1,
              annot=True,
                fmt=".2f"
)
    plt.title("Correlation matrix")
    plt.show()
    return 
    logging.info("Correlation matrix created successfully")

def feature_importance(model, X_train):
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    # A series is a better representation than using a dictionary 
    # index is needed for the labels
    top_features = importances.sort_values(ascending=False)
    # sort first for the importance, then sort again for the visual order of the values
    top_features.sort_values().plot(kind='barh', figsize=(12,6))
    # A horizontal bar chart is much better than a vertical bar chart to represent importance
    plt.title("Top 6 Feature Importances")
    plt.show()
    logging.info("Feature importance created successfully")
  






if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
    correlation_matrix(df)

    tuned_rf = RandomForestClassifier(random_state=42,  
                            class_weight='balanced',
                            max_depth = 17,
                            min_samples_leaf = 5,
                            min_samples_split = 6,
                            n_estimators = 675
                           )
trained_tuned_rf = train_model(tuned_rf, X_train, y_train)
tuned_rf_predictions = make_predictions(trained_tuned_rf, X_test)
feature_importance(trained_tuned_rf, X_train)


# python -m project_1.plots