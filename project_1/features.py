# Checking data types
df.info()

# Missing Values
df.isna().sum()/len(df)

# Correlation matrix analysis
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

# Outliers Analysis quantile method
q1_q = df['X1'].quantile(0.25)
q3_q = df['X1'].quantile(0.75)
# Find the IQR
IQR = q3_q - q1_q
factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
lower_limit_q = q1_q - IQR*factor
upper_limit_q = q3_q + IQR*factor

is_lower_q = df['X1'] < lower_limit_q

is_higher_q = df['X1'] > upper_limit_q
# Combine the masks to filter for outliers
outliers = df['X1'][is_lower_q | is_higher_q] 
# Count and print the number of outliers
print(len(outliers))

q1_q = df['X2'].quantile(0.25)
q3_q = df['X2'].quantile(0.75)
# Find the IQR
IQR = q3_q - q1_q
factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
lower_limit_q = q1_q - IQR*factor
upper_limit_q = q3_q + IQR*factor

is_lower_q = df['X2'] < lower_limit_q

is_higher_q = df['X2'] > upper_limit_q
# Combine the masks to filter for outliers
outliers = df['X2'][is_lower_q | is_higher_q] 
# Count and print the number of outliers
print(len(outliers))

q1_q = df['X3'].quantile(0.25)
q3_q = df['X3'].quantile(0.75)
# Find the IQR
IQR = q3_q - q1_q
factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
lower_limit_q = q1_q - IQR*factor
upper_limit_q = q3_q + IQR*factor

is_lower_q = df['X3'] < lower_limit_q

is_higher_q = df['X3'] > upper_limit_q
# Combine the masks to filter for outliers
outliers = df['X3'][is_lower_q | is_higher_q] 
# Count and print the number of outliers
print(len(outliers))

q1_q = df['X4'].quantile(0.25)
q3_q = df['X4'].quantile(0.75)
# Find the IQR
IQR = q3_q - q1_q
factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
lower_limit_q = q1_q - IQR*factor
upper_limit_q = q3_q + IQR*factor

is_lower_q = df['X4'] < lower_limit_q

is_higher_q = df['X4'] > upper_limit_q
# Combine the masks to filter for outliers
outliers = df['X4'][is_lower_q | is_higher_q] 
# Count and print the number of outliers
print(len(outliers))

q1_q = df['X5'].quantile(0.25)
q3_q = df['X5'].quantile(0.75)
# Find the IQR
IQR = q3_q - q1_q
factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
lower_limit_q = q1_q - IQR*factor
upper_limit_q = q3_q + IQR*factor

is_lower_q = df['X5'] < lower_limit_q

is_higher_q = df['X5'] > upper_limit_q
# Combine the masks to filter for outliers
outliers = df['X5'][is_lower_q | is_higher_q] 
# Count and print the number of outliers
print(len(outliers))

q1_q = df['X6'].quantile(0.25)
q3_q = df['X6'].quantile(0.75)
# Find the IQR
IQR = q3_q - q1_q
factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
lower_limit_q = q1_q - IQR*factor
upper_limit_q = q3_q + IQR*factor

is_lower_q = df['X6'] < lower_limit_q

is_higher_q = df['X6'] > upper_limit_q
# Combine the masks to filter for outliers
outliers = df['X6'][is_lower_q | is_higher_q] 
# Count and print the number of outliers
print(len(outliers))

# Feature engineering is not part of this project as there are only 6 variables and I can not derive anything