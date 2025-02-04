import pandas as pd
import statsmodels.api as sm
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/training_data.csv")
#print(df.head())

Y = df["CO(GT)"]
X = df.drop(columns=["CO(GT)", "AH", "DayOfWeek", "IsWeekend"])
# The columns "AH", "DayOfWeek", and "IsWeekend" we dropped because the p values were > 0.05.

# print(X.head())

X = sm.add_constant(X)

ols_model = sm.OLS(Y, X).fit()

print(ols_model.summary())

# Saving the OLS model to a file
file_path = "data/currentOlsSolution.pkl"

with open(file_path, "wb") as file:
    pickle.dump(ols_model, file)

performance_indicators = {
    "R-squared": ols_model.rsquared,
    "Adjusted R-squared": ols_model.rsquared_adj,
    "F-statistic": ols_model.fvalue,
    "Prob (F-statistic)": ols_model.f_pvalue,
    "AIC": ols_model.aic,
    "BIC": ols_model.bic,
    "Durbin-Watson": sm.stats.durbin_watson(ols_model.resid),
    "Condition Number": ols_model.condition_number
}

performance_df = pd.DataFrame(performance_indicators, index=["OLS model performance"])
#print(performance_df)

performance_df.to_csv("data/OLSModelPerformanceMetrics.csv", header=True, index=True)

# Plotting the diagnostic plots and scatter plots

# Residual plot
plt.figure(figsize=(8, 5))
sns.residplot(x=ols_model.fittedvalues, y=ols_model.resid, lowess=True, line_kws={'color': 'red', 'lw': 2})
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("documentation/OLSTrainingFigures/residual_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Q-Q Plot
sm.qqplot(ols_model.resid, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.savefig("documentation/OLSTrainingFigures/qq_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 5))
sns.histplot(ols_model.resid, kde=True, bins=30)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.savefig("documentation/OLSTrainingFigures/histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Leverage vs. Residuals Plot
fig, ax = plt.subplots(figsize=(10, 5))
sm.graphics.influence_plot(ols_model, ax=ax, criterion="cooks")
plt.title("Influence Plot (Leverage vs. Residuals)")
plt.savefig("documentation/OLSTrainingFigures/leverage_residuals.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter plots
predictors = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)','NO2(GT)', 'PT08.S4(NO2)','PT08.S5(O3)','T', 'RH']
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
axes = axes.flatten()

for i, predictor in enumerate(predictors):
    sns.scatterplot(x=df[predictor], y=df['CO(GT)'], ax=axes[i])
    axes[i].set_xlabel(predictor)
    axes[i].set_ylabel("CO(GT)")
    axes[i].set_title(f"{predictor} vs CO(GT)")

plt.tight_layout()
plt.savefig("documentation/OLSTrainingFigures/scatter_plots.png", dpi=300, bbox_inches='tight')
plt.show()
