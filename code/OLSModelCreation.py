import pandas as pd
import statsmodels.api as sm
import pickle 

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

# with open(file_path, "wb") as file:
#     pickle.dump(ols_model, file)

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