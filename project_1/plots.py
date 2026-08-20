# coorelation matrix

corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr))
correlation_matrix = sns.heatmap(corr, center=0,
            mask=mask,
              linewidths=1,
              annot=True,
                fmt=".2f"
)
plt.savefig("../reports/figures/correlation_matrix.png")
plt.title("Correlation matrix")
plt.show()


# Feature importance

importances = pd.Series(rf_tuned.feature_importances_, index=X_train.columns)
# A series is a better representation than using a dictionary 
# index is needed for the labels
top_6_features = importances.sort_values(ascending=False)
# sort first for the importance, then sort again for the visual order of the values
top_6_features.sort_values().plot(kind='barh', figsize=(12,6))
# A horizontal bar chart is much better than a vertical bar chart to represent importance
plt.title("Top 6 Feature Importances")
plt.savefig("../reports/figures/feature_importance.png")
plt.show()