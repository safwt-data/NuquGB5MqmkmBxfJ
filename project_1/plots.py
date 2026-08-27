import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from project_1.dataset import load_data

def missing_data(df):
   missing = df.isna().sum()/len(df)
   logging.info("Missing data:\n%s", missing)
   return missing


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

def outlier_analysis(df, column):
    q1_q = df[column].quantile(0.25)
    q3_q = df[column].quantile(0.75)
    # Find the IQR
    IQR = q3_q - q1_q
    factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
    lower_limit_q = q1_q - IQR*factor
    upper_limit_q = q3_q + IQR*factor

    is_lower_q = df[column] < lower_limit_q

    is_higher_q = df[column] > upper_limit_q
    # Combine the masks to filter for outliers
    outliers = df[column][is_lower_q | is_higher_q] 
    # Count and print the number of outliers
    logging.info(
        "Number of outliers in %s: %s",
        column,
        len(outliers)
    )
    print(len(outliers))






if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
    missing_data(df)
    correlation_matrix(df)
    outlier_analysis(df, "X1")
    outlier_analysis(df, "X2")
    outlier_analysis(df, "X3")
    outlier_analysis(df, "X4")
    outlier_analysis(df, "X5")
    outlier_analysis(df, "X6")

# python -m project_1.plots