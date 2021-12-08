import pandas as pd


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
print(Stations.head())
Stations = Stations.drop(
    columns=['Station Name', 'Expected Date', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num',
             'EV DC Fast Count', 'EV Other Info', 'EV Network', 'EV Network Web', 'Date Last Confirmed',
             'EV Connector Types', 'Facility Type', 'EV Pricing'])

EV_chargingpoints = Stations[Stations["Fuel Type Code"] == "ELEC"]
EV_chargingpoints = EV_chargingpoints[EV_chargingpoints['Country'] == 'US']
EV_chargingpoints = EV_chargingpoints[EV_chargingpoints['Access Code'] == 'public']
# EV_chargingpoints = EV_chargingpoints[EV_chargingpoints['State'] == 'CA']


EV_chargingpoints['Open Date'] = pd.to_datetime(EV_chargingpoints['Open Date'], format='%Y-%m-%d')
EV_chargingpoints['Year-Month'] = EV_chargingpoints['Open Date'].dt.strftime('%Y-%m')
EV_chargingpoints = EV_chargingpoints.drop(columns=['Fuel Type Code', 'Open Date', 'Country', 'Access Code', 'State'])
EV_chargingpoints['Chargepoints'] = 1
EV_chargingpoints = EV_chargingpoints.groupby(['Year-Month']).sum()

print(EV_chargingpoints.index)
print(EV_chargingpoints.index)

print(type(EV_chargingpoints.index))
EV_chargingpoints.index = pd.DatetimeIndex(EV_chargingpoints.index).to_period('M')

print(EV_chargingpoints.index)
print(type(EV_chargingpoints.index))
