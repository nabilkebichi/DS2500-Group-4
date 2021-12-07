#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas


# ### Create master state DataFrame with:
# Median income, EV registrations per capita, total population, and geographic coordinates for mapping

# In[2]:


# census data by region
# https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-income-households.html
us = pd.read_excel('median_income_region.xlsx', header=(7, 8))
northeast = pd.read_excel('median_income_region.xlsx',
                          'Northeast', header=(1, 2))
midwest = pd.read_excel('median_income_region.xlsx', 'Midwest', header=(1, 2))
south = pd.read_excel('median_income_region.xlsx', 'South', header=(1, 2))
west = pd.read_excel('median_income_region.xlsx', 'West', header=(1, 2))


# In[3]:


# pull 2020 median income by region into new list
median_2020 = []

median_2020.append(us['Median income']['2020\ndollars'][0])
median_2020.append(northeast['Median income']['2020\ndollars'][0])
median_2020.append(midwest['Median income']['2020\ndollars'][0])
median_2020.append(south['Median income']['2020\ndollars'][0])
median_2020.append(west['Median income']['2020\ndollars'][0])

median_2020


# In[4]:


# new df with region and 2020 median income data
median_income_region = pd.DataFrame()

regions = ['United States', 'Northeast', 'Midwest', 'South', 'West']

median_income_region['Regions'] = regions
median_income_region['Median Income 2020 (2020 Dollars)'] = median_2020

median_income_region


# In[5]:


plt.figure(figsize=(6, 6))
plt.bar(regions, median_2020)
plt.title('2020 Median Household Income by Region')
plt.xlabel('Region')
plt.ylabel('Income (2020 dollars)')
plt.show


# In[6]:


# census data by state
# https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-income-households.html

state_median = pd.read_excel('median_income_state.xlsx', header=(62, 63))

state_median.head(1)


# In[7]:


# clean headers and remove duplicate years
headers = []
drop = []

for year in range(1984, 2021):
    headers.append(year)

for header in state_median:
    if type(header[0]) == str and header[0] != 'State':
        new_header = int(header[0][:4])

        if new_header in headers:
            drop.append(header)
            state_median.drop([header], axis=1, inplace=True)
        else:
            state_median.rename(columns={header[0]: new_header}, inplace=True)


# In[8]:


state_median.head()


# In[9]:


# new df with states and 2020 median income data
states = state_median['State']['Unnamed: 0_level_1']
median2020 = state_median[2020]['Median income']
frame = {'State': states, '2020 Median Income': median2020}


# In[10]:


state_median2020 = pd.DataFrame(frame)
state_median2020.head()


# In[11]:


# data on EV registrations by state
# https://afdc.energy.gov/data/10962
evreg_state = pd.read_excel('10962-ev-registration-counts-by-state_6-11-21.xlsx',
                            header=(2), usecols=['State', 'Registration Count'])
evreg_state.rename(
    columns={"Registration Count": "EV Registration Count"}, inplace=True)

evreg_state.head()


# In[12]:


# merge 2020 income and EV registration dfs
state_info = pd.merge(state_median2020, evreg_state, on='State')
states = geopandas.read_file('shapes/shapefile.shx')
states = states[['NAME', 'STUSPS', 'geometry']]
states = states.rename(columns={'NAME': 'State'})

state_geo = pd.merge(states, state_info, on='State')

# print(state_geo.head())
state_info.head()

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.axis('off')
ax.set_title('Heat Map of Median Income (USA)', fontdict={
             'fontsize': '19', 'fontweight': '2'})

color = 'Oranges'
vmin, vmax = 0, 231
sm = plt.cm.ScalarMappable(
    cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
cbar.ax.tick_params(labelsize=10)

state_geo.plot('2020 Median Income', cmap=color, linewidth=0.8,
               ax=ax, edgecolor='0.8', figsize=(10, 10))


# In[13]:


# census data on state populations in 2019 (most recent year)
# https://data.census.gov/cedsci/table?q=Total%20Population&g=0100000US%240400000&tid=ACSDP1Y2019.DP05
state_pop = pd.read_excel('ACSDP1Y2019.DP05-2021-12-05T182017.xlsx', 'Data')

# remove repeat instances of Total population LI by taking only top 5 rows
state_pop = state_pop.head(5)
state_pop = state_pop.T  # transpose data to match merged df
new_header = state_pop.iloc[0]  # take first row to be new header
state_pop = state_pop[1:]  # new data minus first row (new header)
state_pop.columns = new_header
state_pop.drop(['Label', 'SEX AND AGE', 'Male', 'Female'],
               axis=1, inplace=True)  # drop all irrelevent data

state_pop.head()


# In[14]:


state_pop.reset_index(inplace=True)
# rename columns to match naming conventions
state_pop.rename(columns={"index": "State"}, inplace=True)
state_pop.head()


# In[15]:


# remove columns, str to int
state_pop['Total population'] = state_pop['Total population'].str.replace(
    ',', '').astype(int)


# In[16]:


# add population data to master state df
state_info = pd.merge(state_info, state_pop, on='State')
state_info.head()


# In[17]:


# calculated field for EV registrations per capita to add context to numbers
state_info['EV Registations per 1000'] = state_info['EV Registration Count'] / \
    state_info['Total population'] * 1000


# In[18]:


# import state coordinates from tsv file
# https://www.census.gov/geographies/reference-files/2010/geo/state-area.html

state_coord_area = pd.read_csv('census_state_data.txt', sep='\t', names=['State', 'Square Miles', 'total Sq. Km.', 'land Sq. Mi.',
                                                                         'land Sq. Km.', 'water Sq. Mi.', 'water Sq. Km.', 'inland Sq. Mi.',
                                                                         'inland Sq. Km.', 'coastal Sq. Mi.', 'coastal Sq. Km.', 'gl Sq. Mi.',
                                                                         'gl Sq. Km.', 'territorial Sq. Mi.', 'territorial Sq. Km.', 'Latitude', 'Longitude'])

state_coord_area.head()


# In[19]:


state_coord_area.drop(['total Sq. Km.', 'land Sq. Mi.', 'land Sq. Km.', 'water Sq. Mi.',
                       'water Sq. Km.', 'inland Sq. Mi.', 'inland Sq. Km.', 'coastal Sq. Mi.',
                       'coastal Sq. Km.', 'gl Sq. Mi.', 'gl Sq. Km.', 'territorial Sq. Mi.', 'territorial Sq. Km.'],
                      axis=1, inplace=True)


# In[20]:


# add state coordinates to master state df
state_info = pd.merge(state_info, state_coord_area, on='State')


# In[21]:


state_info['Square Miles'] = state_info['Square Miles'].str.replace(
    ',', '').astype(int)


# In[22]:


state_info['EV Registations per sq mile'] = state_info['EV Registration Count'] / \
    state_info['Square Miles']


# In[23]:


state_info.head(1)


# In[24]:


# for graphing, we need to drop Alaska, Hawaii, and DC
continental_state_info = state_info.drop([1, 8, 11])
continental_state_info


# In[25]:


plt.figure(figsize=(10, 10))
sns.barplot(x='State', y='EV Registations per sq mile',
            data=continental_state_info)


# In[26]:


# New function to map temperatures to a 2D Numpy array
# based on the GPS cordinates of the relevent station.
# Numpy array is 150 x 100 to reflect range in Latitude
# and Longitude

def create_map(input_df, measure):

    x_dimen = 155
    y_dimen = 100

    empty_map = np.zeros((y_dimen, x_dimen), dtype=int)

    longitudes = []
    latitudes = []
    data = []

    for index, row in input_df.iterrows():
        longitudes.append(row['Longitude'])
        latitudes.append(row['Latitude'])
        data.append(row[measure])

    for i in range(len(data)):
        lon_percent = (longitudes[i]-(-125.0)) / (-65-(-125.0))
        lat_percent = (latitudes[i]-(50.0)) / (25.0-(50.0))

        x_cord = int(lon_percent * 150)
        y_cord = int(lat_percent * 100)

        # automatically make values <1 = 1 for graphing purposes
        if data[i] > 1:
            empty_map[y_cord][x_cord] = data[i]
        if data[i] < 1:
            empty_map[y_cord][x_cord] = 1

    return empty_map


# In[66]:


def size_mapping(input_array):

    x_dimen = 155
    y_dimen = 100
    z_dimen = 3

    heat_map = np.zeros((y_dimen, x_dimen, z_dimen), dtype=int)

    white = [255, 255, 255]
    blue = [0, 0, 204]
    red = [204, 0, 0]

    x = 0
    for lat in input_array:
        y = 0
        for ev in lat:
            if ev == 0:
                heat_map[x][y] = white

            size = ev

            # if the value == 1 then color blue, otherwise color red
            if ev == 1:
                # THIS CODE DOES WORK - NEED TO FIX
                for i in range(-size, (size+1)):
                    for j in range(-size, (size+1)):
                        heat_map[x+i][y+j] = blue

            if ev != 0:
                if ev != 1:
                    for i in range(-size, (size+1)):
                        for j in range(-size, (size+1)):
                            heat_map[x+i][y+j] = red

            y += 1
        x += 1

    return heat_map


# In[67]:


def create_image(map_data):

    plt.figure(figsize=(12, 12))
    plt.imshow(map_data, interpolation="none")


# In[68]:


create_image(size_mapping(create_map(
    continental_state_info, 'EV Registations per sq mile')))


# In[69]:


create_image(size_mapping(create_map(
    continental_state_info, 'EV Registations per 1000')))


# In[ ]:
