import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings

# Sets Pandas options to view the whole dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Import Data, and drop most columns.  All we want is the dates, and to make sure it's an EV station
Stations = pd.read_csv("alt_fuel_stations (Dec 7 2021).csv", encoding='utf-8')
Stations = Stations.drop(
    columns=['Street Address', 'Intersection Directions', 'City', 'ZIP', 'Plus4', 'Station Phone', 'Status Code',
             'Groups With Access Code', 'Access Days Time', 'Cards Accepted', 'BD Blends', 'NG Fill Type Code',
             'NG PSI', 'Geocode Status', 'Latitude', 'Longitude', 'ID', 'Updated At', 'Owner Type Code',
             'Federal Agency ID', 'Federal Agency Name', 'Hydrogen Status Link', 'NG Vehicle Class', 'LPG Primary',
             'E85 Blender Pump', 'Intersection Directions (French)', 'Access Days Time (French)', 'BD Blends (French)',
             'Groups With Access Code (French)', 'Hydrogen Is Retail', 'Access Detail Code',
             'Federal Agency Code', 'EV Pricing (French)', 'LPG Nozzle Types', 'Hydrogen Pressures',
             'Hydrogen Standards', 'CNG Fill Type Code', 'CNG PSI', 'CNG Vehicle Class', 'LNG Vehicle Class',
             'EV On-Site Renewable Source', 'Restricted Access', 'CNG Dispenser Num', 'CNG On-Site Renewable Source',
             'CNG Total Compression Capacity', 'CNG Storage Capacity', 'LNG On-Site Renewable Source',
             'E85 Other Ethanol Blends'])

Stations = Stations.drop(
    columns=['Station Name', 'Expected Date', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num',
             'EV DC Fast Count', 'EV Other Info', 'EV Network', 'EV Network Web', 'Date Last Confirmed',
             'EV Connector Types', 'Facility Type', 'EV Pricing'])

# Some basic filtering. We only want US data, public access ports, and obviously EV stations. Optional filter for state
EV_chargingpoints = Stations[Stations["Fuel Type Code"] == "ELEC"]
EV_chargingpoints = EV_chargingpoints[EV_chargingpoints['Country'] == 'US']
EV_chargingpoints = EV_chargingpoints[EV_chargingpoints['Access Code'] == 'public']
# EV_chargingpoints = EV_chargingpoints[EV_chargingpoints['State'] == 'CA']

# Convert Station open date to a Year-Month string, and creates new column Year-Month
EV_chargingpoints['Open Date'] = pd.to_datetime(EV_chargingpoints['Open Date'], format='%Y-%m-%d')
EV_chargingpoints['Year-Month'] = EV_chargingpoints['Open Date'].dt.strftime('%Y-%m')

# Drops other last set of filter columns. We are just left with Year-Month, also adds Chargepoints, the column we
# will sum() on when we group by Year-Month
EV_chargingpoints = EV_chargingpoints.drop(columns=['Fuel Type Code', 'Open Date', 'Country', 'Access Code', 'State'])
EV_chargingpoints['Chargepoints'] = 1
EV_chargingpoints = EV_chargingpoints.groupby(['Year-Month']).sum()

# Drops this random future chargepoint.  Data quality for this analysis isn't great
EV_chargingpoints = EV_chargingpoints.drop(labels=['2022-06'], axis=0)

# Creates and adds the 2022 year to the Dataframe.  0's are reported as these are the values we will be modeling
EV_2022 = pd.DataFrame([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], columns=['Chargepoints'],
                       index=['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08',
                              '2022-09', '2022-10', '2022-10', '2022-12'])
EV_chargingpoints = EV_chargingpoints.append(EV_2022)

# Set standards for the first plt.plot(), the reflection of all data
sns.set()
plt.tick_params(labelbottom=False, bottom=False)
plt.xlabel('1995 - 2022 (Monthly)')
plt.ylabel('EV Stations Opened')
plt.title("EV Charging Station Growth")

# Plots and Shows the charging station built per month over time
plt.plot(EV_chargingpoints.index, EV_chargingpoints['Chargepoints'], )
plt.show()

# Creates training and testing datasets for ARIMA Estimator Model
train = EV_chargingpoints[EV_chargingpoints.index <= '2020-01']
test = EV_chargingpoints[EV_chargingpoints.index > '2020-01']

# Creates and fits ARIMA Model. Some other models we found that worked were,(2, 2, 1), (8, 3, 3).  Slight changes to
# the data drastically affect the consistency of the ARIMA Model
y = train['Chargepoints']
ARIMAmodel = ARIMA(y, order=(8, 3, 2))
# ARIMAmodel = ARIMA(y, order = (2, 2, 1))
# ARIMAmodel = ARIMA(y, order = (8, 3, 3))
# ARIMAmodel = ARIMA(y, order=(2, 3, 1))
# ARIMAmodel = ARIMA(y, order=(4, 3, 1))

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
