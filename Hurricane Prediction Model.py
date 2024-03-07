import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

hurricane = pd.read_csv("storms.csv")
hurricane

hurricane.apply(pd.isnull)

hurricane["category"].value_counts()
hurricane["category"] = hurricane["category"].fillna(0)
hurricane["tropicalstorm_force_diameter"] = hurricane["tropicalstorm_force_diameter"].fillna(0)
hurricane["hurricane_force_diameter"] = hurricane["hurricane_force_diameter"].fillna(0)
hurricane

hurricane['Datatime'] = pd.to_datetime(hurricane[['year', 'month', 'day', 'hour']])
hurricane = hurricane.drop(columns=['year', 'month', 'day', 'hour'])
hurricane = hurricane[['Datatime'] + [col for col in hurricane if col != 'Datatime']]
hurricane = hurricane.drop(hurricane.columns[1], axis=1)
hurricane.set_index('Datatime', inplace=True)
hurricane

core_hurricane = hurricane[["lat", "long", "wind", "pressure"]].copy()
core_hurricane.columns = ["Latitude","Longitude", "Wind_speed", "Pressure"]
core_hurricane

core_hurricane.apply(pd.isnull).sum()/core_hurricane.shape[0]
core_hurricane.dtypes
core_hurricane.index
core_hurricane.index.year.value_counts().sort_index()
core_hurricane["Latitude"].plot()

core_hurricane["Target"] = core_hurricane.shift(-1)["Latitude"]
core_hurricane = core_hurricane.iloc[:-1,:].copy()
core_hurricane

from sklearn.linear_model import Ridge
reg = Ridge(alpha=.1)
predictors = ["Latitude", "Longitude", "Wind_speed", "Pressure"]
train = core_hurricane.loc[:"2010-11-10"]
test = core_hurricane.loc["2011-06-28":]

reg.fit(train[predictors], train["Target"])
predictions = reg.predict(test[predictors])
from sklearn.metrics import mean_absolute_error
mean_absolute_error(test["Target"], predictions)

combined = pd.concat([test["Target"], pd.Series(predictions, index = test.index)], axis=1)
combined.columns = ["actual", "predictions"]
combined

combined.plot()

reg.coef_

def create_predictions(predictors, core_hurricane, reg)
    train = core_hurricane.loc[:"2010-11-10"]
    test = core_hurricane.loc["2011-06-28":]
    reg.fit(train[predictors], train["Target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error((test["Target"], predictions))
    combined = pd.concat([test["Target"], pd.Series(predictions, index = test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined

core_hurricane["month_max"] = core_hurricane["Latitude"].rolling(30).mean()
core_hurricane["month_day_max"] = core_hurricane["month_max"] / core_hurricane["Latitude"]
#core_hurricane["max_min"] = core_hurricane["Latitude"] / core_hurricane["Longitude"]
predictors = ["Latitude", "Longitude", "Wind_speed", "Pressure", "month_max", "month_day_max"]
core_hurricane = core_hurricane.iloc[30:,:].copy()
error, combined = create_predictions(predictors, core_hurricane, reg)
error


 