import re, io, os, sys, ast, ssl, base64, csv, json, sqlite_utils, requests
from datetime import datetime as DT, timezone as TZ, timedelta as TD
from sqlite_utils.utils import sqlite3

import pycountry
from deep_translator import GoogleTranslator
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapi")

async def crtCTYdb():
  for country in pycountry.countries:
    ctyName = country.name
    
    if hasattr(country, 'official_name'):
      offset_name = country.official_name
    else:
      offset_name = 'N/A'
      
    translated_ctyName = GoogleTranslator(source='auto', target='zh-TW').translate(ctyName)
    
    location = geolocator.geocode(country.name)
    if location:
      lat = location.latitude
      lon = location.longitude
    else:
      lat = ''
      lon = ''

    jsonData = {
      'NUMERIC': str(country.numeric),
      'NAME': str(country.name),
      'NAME_ZH': str(translated_ctyName),
      'OFFSET_NAME': str(offset_name),
      'ALPHA_2':str(country.alpha_2),
      'ALPHA_3':str(country.alpha_3),
      'lat': str(lat),
      'lon': str(lon),
      'desc_': '',
    }
    
    db = 'country.db3'
    conn = sqlite_utils.Database(db)
    try:
      conn['CTYLST'].insert(jsonData, pk=('NUMERIC'), alter=True)
      print(f'Insert {country.name}')
    except sqlite3.IntegrityError as e:
      conn['CTYLST'].update(jsonData['NUMERIC'], jsonData, alter=True)
      print(f'Update {country.name}')
