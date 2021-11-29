import pandas as pd
import re

pd.set_option('display.max_columns', None)

"""
Data Source : https://data.world/data-society/used-cars-data
"""


def snip_year(text):
    # text = re.sub(r"[0-9]{2}\-[0-9]{2}", "this", text)
    text = re.findall(r"^[0-9]{4}", text)
    return int(text[0])


def snip_month(text):
    text = re.findall(r"\-([0-9]{2})\-", text)
    return int(text[0])


def snip_day(text):
    text = re.findall(r"\-([0-9]{2})\b", text)
    return int(text[1])


# print(snip_month('2016-03-24 11:52:17'))


used_car_df = pd.read_csv("datasets/autos.csv", encoding='latin-1')
used_car_df = used_car_df.drop(
    columns=['seller', 'offerType', 'abtest', 'monthOfRegistration', 'nrOfPictures', 'postalCode', "dateCrawled",
             'name'])

used_car_df['Sold_Year'] = used_car_df['lastSeen'].apply(snip_year)
used_car_df['Sold_Month'] = used_car_df['lastSeen'].apply(snip_month)
used_car_df['Sold_Day'] = used_car_df['lastSeen'].apply(snip_day)

used_car_df = used_car_df.reindex(
    columns=['yearOfRegistration', 'brand', 'model', 'vehicleType', 'fuelType', 'gearbox', 'kilometer', 'price',
             'powerPS', 'notRepairedDamage', 'dateCreated', 'lastSeen', 'Sold_Year', 'Sold_Month', 'Sold_Day'])

col_list = []
for col in used_car_df:
    col_list.append(col)

print(col_list)
print(used_car_df.head(),used_car_df.shape)
