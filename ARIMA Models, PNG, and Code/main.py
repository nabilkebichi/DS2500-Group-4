import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings

# Each model I test throws 6 warnings, due to datatype and some early indexing issues. I test 64 models so this adds up
warnings.filterwarnings("ignore")

# Sets Pandas options to view the whole dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def read_n_clean_1(location):
    # Import Data, and drop most columns
    df = pd.read_csv(location, encoding='utf-8')
    df = df.drop(
        columns=['Street Address', 'Intersection Directions', 'City', 'ZIP', 'Plus4', 'Station Phone', 'Status Code',
                 'Groups With Access Code', 'Access Days Time', 'Cards Accepted', 'BD Blends', 'NG Fill Type Code',
                 'NG PSI', 'Geocode Status', 'Latitude', 'Longitude', 'ID', 'Updated At', 'Owner Type Code',
                 'Federal Agency ID', 'Federal Agency Name', 'Hydrogen Status Link', 'NG Vehicle Class', 'LPG Primary',
                 'E85 Blender Pump', 'Intersection Directions (French)', 'Access Days Time (French)',
                 'BD Blends (French)',
                 'Groups With Access Code (French)', 'Hydrogen Is Retail', 'Access Detail Code',
                 'Federal Agency Code', 'EV Pricing (French)', 'LPG Nozzle Types', 'Hydrogen Pressures',
                 'Hydrogen Standards', 'CNG Fill Type Code', 'CNG PSI', 'CNG Vehicle Class', 'LNG Vehicle Class',
                 'EV On-Site Renewable Source', 'Restricted Access', 'CNG Dispenser Num',
                 'CNG On-Site Renewable Source',
                 'CNG Total Compression Capacity', 'CNG Storage Capacity', 'LNG On-Site Renewable Source',
                 'E85 Other Ethanol Blends'])
    return df


def read_n_clean_2(df):
    # Some basic filtering. We only want US data, public access ports, and obviously EV stations. Optional filter for
    # state
    df = df.drop(
        columns=['Station Name', 'Expected Date', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num',
                 'EV DC Fast Count', 'EV Other Info', 'EV Network', 'EV Network Web', 'Date Last Confirmed',
                 'EV Connector Types', 'Facility Type', 'EV Pricing'])

    df = df[df["Fuel Type Code"] == "ELEC"]
    df = df[df['Country'] == 'US']
    df = df[df['Access Code'] == 'public']
    return df


def format_and_summarize(df):
    # Convert Station open date to a Year-Month string, and creates new column Year-Month
    df['Open Date'] = pd.to_datetime(df['Open Date'], format='%Y-%m-%d')
    df['Year-Month'] = df['Open Date'].dt.strftime('%Y-%m')

    # Drops other last set of filter columns. We are just left with Year-Month, also adds Chargepoints, the column we
    # will sum() on when we group by Year-Month
    df = df.drop(columns=['Fuel Type Code', 'Open Date', 'Country', 'Access Code', 'State'])
    df['Chargepoints'] = 1
    df = df.groupby(['Year-Month']).sum()
    return df


def create_model_maker(min, max):
    # Creates a multi nested list for use in ARIMA estimation Models
    m_list = []
    for a in range(min, max + 1):
        for b in range(min, max + 1):
            for c in range(min, max + 1):
                m_list.append([0, [a, b, c]])
    return m_list


def test_models(model_list, train, test):
    # Runs all possible models from model_list.  Scores them on RMSE, and sorts to identify the best model for data
    # model_list: a custom nested list full of models. Created from create_model_maker()
    # train: Training dataset, index and values
    # test: Scoring dataset, index and values

    for model in range(len(model_list)):
        aa = model_list[model][1][0]
        bb = model_list[model][1][1]
        cc = model_list[model][1][2]

        ARIMAmodel = ARIMA(train['Chargepoints'], order=(aa, bb, cc))
        ARIMAmodel = ARIMAmodel.fit()
        y_pred = ARIMAmodel.get_forecast(len(test.index))
        y_pred_df = y_pred.conf_int(alpha=0.05)
        y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df.index = test.index
        arma_rmse = np.sqrt(mean_squared_error(test["Chargepoints"].values, y_pred_df["Predictions"]))
        model_list[model][0] = arma_rmse
        print(model_list[model])
        print('Running ARIMA Model # ', model + 1, 'of ', len(model_list))
    model_list.sort(key=lambda x: x[0], reverse=False)
    return model_list


def test_train_date(df, date):
    # Creates training and testing datasets for ARIMA Estimator Model"""
    train = df[df.index <= date]
    test = df[df.index > date]
    return train, test


def print_base_graph(df, filt):
    # Set standards for the first plt.plot(), the reflection of all data
    sns.set()
    plt.tick_params(labelbottom=False, bottom=False)
    plt.xlabel('1995 - 2021 (Monthly)')
    plt.ylabel('EV Stations Opened each Month')
    plt.title("EV Charging Station Growth -> Filters: " + filt)

    # Plots and Shows the charging station built per month over time
    plt.plot(df.index, df['Chargepoints'])
    plt.show()


def add_2022(df):
    # Creates and adds the 2022 year to the Dataframe.
    # 0's are reported as these are the values we will be estimating with our newly found optimal model
    df2 = pd.DataFrame([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], columns=['Chargepoints'],
                       index=['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08',
                              '2022-09', '2022-10', '2022-10', '2022-12'])
    df = df.append(df2)
    return df


def final_graph(train, model_lst, state_filter):
    # Creates final Graph with optimal model

    # Inputs optimal ARIMA model and fits it to the data
    ARIMAmodel = ARIMA(train['Chargepoints'], order=(model_lst[0][1][0], model_lst[0][1][1], model_lst[0][1][2]))
    ARIMAmodel = ARIMAmodel.fit()

    # The guts of the procedure. Applys model with confidence level of .05,
    y_pred = ARIMAmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Predictions"]

    # Set standards for the second plt.plot(), with 3 different data sets, the Train, Test, and Estimate
    sns.set()
    plt.tick_params(labelbottom=False, bottom=False)
    plt.xlabel('1995 - 2022 (Monthly)')
    plt.ylabel('EV Stations Opened Each Month')
    plt.title("ARIMA Model Estimates for Charging Station Growth in " + state_filter)

    # Plots and shows the 3 data sets, the Train, Test, and Estimate
    plt.plot(train, color="black", label='1995-2020 Data (Train)')
    plt.plot(test, color="red", label='2020-2021 Data (Test)')
    plt.plot(y_pred_out, color='Green', label='ARIMA Estimate ' + str(omni[0][1]))
    plt.legend()
    plt.show()


"""----Intro Data----"""
Stations = read_n_clean_1("alt_fuel_stations (Dec 7 2021).csv")
Stations = read_n_clean_2(Stations)

state_filter = 'CO'
EV_chargingpoints = Stations[Stations['State'] == state_filter]
EV_chargingpoints = format_and_summarize(EV_chargingpoints)

"""----Start Graph 1----"""

print_base_graph(EV_chargingpoints, 'State of ' + state_filter)

"""----Creation & Curation of ARIMA Models----"""
# Creation of Test and Train Datasets,
train, test = test_train_date(EV_chargingpoints, '2020-01')

# Creation of the model training list
omni = create_model_maker(0, 3)

# Run models in list and discover the best model, via RSME
omni = test_models(omni, train, test)

"""----Start Graph 2----"""
# Adds in 2022 blanks for model to estimate
EV_chargingpoints = add_2022(EV_chargingpoints)

# re-creates training and testing datasets for ARIMA Estimator Model, now with 2022 blanks
train, test = test_train_date(EV_chargingpoints, '2020-01')

final_graph(train, omni, state_filter)
