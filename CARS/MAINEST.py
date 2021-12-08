
''''''
# ARIMAmodel = ARIMA(y, order=(8, 3, 2))
### ARIMAmodel = ARIMA(y, order = (2, 2, 1))
### ARIMAmodel = ARIMA(y, order = (8, 3, 3))
### ARIMAmodel = ARIMA(y, order=(2, 3, 1))
### ARIMAmodel = ARIMA(y, order=(4, 3, 1))

ARIMAmodel = ARIMAmodel.fit()

"""
ARIMA Model code source: https://builtin.com/data-science/time-series-forecasting-python
and https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
"""

# Long story short, creates the model, sets the confidence level, and then runs the model off of the DF index
# Creates y_pred_out which are the Model's predictions, to be graphed
y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha=0.05)
y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"]

"""
This can be seen graphically, but our model predicts EV station creation at the rate of 
479 - 627 per month for 2020
592 - 786 per month for 2021
801 - 974 per month for 2022

via print(y_pred_out)
"""

# Set standards for the second plt.plot(), with 3 different data sets, the Train, Test, and Estimate
sns.set()
plt.tick_params(labelbottom=False, bottom=False)
plt.xlabel('1995 - 2022 (Monthly)')
plt.ylabel('EV Stations Opened')
plt.title("Train/Test split for Charging data")

# Plots and shows the 3 data sets, the Train, Test, and Estimate
plt.plot(train, color="black", label='1995-2020 Data (Train)')
plt.plot(test, color="red", label='2020-2020 Data (Test)')
plt.plot(y_pred_out, color='Green', label='ARIMA Estimate (8, 3, 2)')
plt.legend()

plt.show()

# For model evaluation, we used a combination of visual and numerical process, relying on our graph and the RMSE of
# the produced model, working to create a model that was both graphically and statistically relevant
arma_rmse = np.sqrt(mean_squared_error(test["Chargepoints"].values, y_pred_df["Predictions"]))
print("RMSE: ", arma_rmse)

# This will create index warnings however the code runs smoothly. It is simply from the adaptations I made from the
# source.  Our data is formatted quite differently and the source doesn't have groups or summarized data. Thanks!
