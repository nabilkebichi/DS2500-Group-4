import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import geopandas as gpd
# function ot read data from csv file
def read_data(file_name):
    df = pd.read_csv(file_name)
    return df


# main function
def main():

    df = read_data('incentives.csv')
    # select rows where fuel type code is ELEC

    print (df.head())
    plt.figure(figsize=(12, 5))

    # count number of rows for each state in the dataframe
    df = df.groupby('State').size()
    df.sort_values(ascending=False, inplace=True)

    df.plot(kind='bar')
    plt.title('EV Incentives by State')
    plt.ylabel('Number of EV Incentives')
    plt.xlabel('State')
    plt.bar(df.index, df.values, width=0.5)
    plt.show()


main()