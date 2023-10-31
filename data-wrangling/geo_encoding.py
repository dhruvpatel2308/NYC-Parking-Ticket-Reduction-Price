import pandas as pd
import geopy
from geopy.geocoders import ArcGIS
import requests
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../Dataset/parking_violation.csv')

# Combine street_name along with state to create proper address
df['address'] = df['street_name'].str.cat(others=df[['state']], sep=",", na_rep="")

unq_add = df['address'].drop_duplicates()

def clean_street_name(data):
  """
  This function removes unwanted diretional information from street name to make it more
  clear to find its co-ordinates.
  Arguments:
  data: a string that will contain the street_name
  Returns:
  string with just the street name
  """
  filters = [
      " (S/B) @",
      " (S/B)",
      " (S/B",
      " (S/",
      " (S",

      " (E/B) @",
      " (E/B)"
      " (E/B"
      " (E/",
      " (E"

      " (N/B) @",
      " (N/B)",
      " (N/B",
      " (N/",
      " (N",

      " (W/B) @",
      " (W/B)",
      " (W/B",
      " (W/",
      " (W",

      " (",
      "@"
  ]
  
  for filter in filters:
    data = data.replace(filter, "")

  return data

# Perform the cleaning of street name
df['address'] = df['address'].apply(lambda x: clean_street_name(x))

# Create a geolocator object to fetch co-ordinate details based on address
geolocator_arcis = ArcGIS()

def geolocate_data(addresses, geoloc):
  """
  For given address find the co-ordinate for all those possible.
  Arguments:
  addresses: List of strings usually containing address of street,
  geoloc: A geo-locator object to fetch co-ordinates
  Returns:
  a dictonary of each street and their coresponding latitude and logitude
  """

  geoencode_add = []
  MAX_SKIP = 10
  skip_cnt = 0

  for idx, add in enumerate(addresses):
    try:
      location = geoloc.geocode(add)

      if location is not None:
        genc_data = {
            'street_name': add,
            'lat': location.latitude,
            'lon': location.longitude
        }
        geoencode_add.append(genc_data)
    except:
      skip_cnt += 1
      if skip_cnt > MAX_SKIP:
        break
      print("Skiping for ", add, " - ", type(add))
      continue
    
    # Checkpoint to keep track of how much address we processed and out of that
    # for how many address we got co-ordinate information
    if (idx+1) % 200 == 0:
      print(f"{idx+1} Address completed, and fetched {len(geoencode_add)}")
  
  geo_encode_dict = {}

  for item in geoencode_add:
    geo_encode_dict[item['street_name']] = {
        'lat': item['lat'],
        'lon': item['lon']
    }
  return geo_encode_dict

# This function will try to find co-oridnate for address 
geo_encode_dict = geolocate_data(list(df['address']), geolocator_arcis)

# Store the latitude and logitude of the streets for which it is available
df['lat'] = df['address'].apply(lambda x: geo_encode_dict[x]['lat'] if x in geo_encode_dict else None)
df['lon'] = df['address'].apply(lambda x: geo_encode_dict[x]['lon'] if x in geo_encode_dict else None)

# We have a Violation County with redudant values so setting it to common names which is useful
# There are 5 Counties in New-York city names
# 1. Brooklyn
# 2. Queens
# 3. Manhattan
# 4. Staten Island
# 5. Bronx

remap_county_dict = {
    'K' : 'Brooklyn',
    'Q' : 'Queens',
    'NY': 'Manhattan',
    'QN': 'Queens',
    'BK': 'Brooklyn',
    'R' : 'Staten Island',
    'BX': 'Bronx',
    'ST': 'Staten Island',
    'MN': 'Manhattan',
    'KINGS': 'Brooklyn',
    'QNS': 'Queens',
    'BRONX': 'Bronx'
}
df['Violation County'] = df['county'].map(remap_county_dict).astype('category')
df['Violation County'].value_counts()