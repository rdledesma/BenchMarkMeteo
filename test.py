import splitting_models.models as sm
import pandas as pd
import matplotlib.pyplot as plt
import os
from Geo import Geo
from Sites import Site
import Metrics
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
import Metrics as m
site = Site('ERO')


site.describe()

data = pd.read_csv(f'/home/inenco/Documentos/Medidas/{str.lower(site.cod)}_15.csv')

data['datetime'] = pd.to_datetime(data.datetime)

data = (data.set_index('datetime')
      .reindex(pd.date_range(
    start="2013/01/01 00:00", end="2024/12/31 23:59", freq="15min"))
      .rename_axis(['datetime'])
      .reset_index())



plt.figure()
plt.plot(data.datetime, data.ghi, label="Meas")
plt.show(block=False)

cams = pd.read_csv(f'/home/inenco/Documentos/HELIOSAT/{str.lower(site.cod)}.csv', usecols=['GHI','Clear sky GHI','BNI'], sep=";", 
                   header=42)

cams['datetime'] = pd.date_range(
    start="2013/01/01 00:00", end="2024/12/31 23:59", freq="15min"
)

cams['GHI'] = cams.GHI * 4
cams['GHIcs'] = cams['Clear sky GHI'] * 4
cams['DNI'] = cams['BNI'] * 4


cams.columns
sm.get('Yang5').__required__

from datetime import timedelta

dfGeo = Geo(range_dates=data.datetime + timedelta(minutes=7.5),
            lat=site.lat, long=site.long, alt=site.alt, gmt=0, beta=0).df

data['SZA'] = dfGeo.SZA.values
data['TOA'] = dfGeo.TOA.values
data['GHI'] = cams.GHI.values
data['GHIcsk'] = dfGeo.GHIargp2.values
data['DNI'] = cams.DNI.values
data['longitude'] = site.long
data['eth'] = data.TOA
data['sza'] = data.SZA
data['ghics'] = data.GHIcsk
data['latitude'] = site.lat
data["timestamp"] = pd.to_datetime(data["datetime"], utc=True)

data = data.resample(on='timestamp', rule="60 min").mean().reset_index()

data = data.set_index("timestamp")





inp = data[["latitude","longitude", "eth", "sza", "ghi", "ghics"]].copy()
pred = sm.get("Yang5").predict(inp)



inp = data[["latitude","longitude", "eth", "sza", "GHI", "ghics"]].copy()
inp.columns = ['latitude', 'longitude', 'eth', 'sza', 'ghi', 'ghics']
pred2 = sm.get("Yang5").predict(inp)



data['Delta'] = data.ghi - data.ghics

X = data.dropna()
X = X[X.datetime.dt.year == 2015]
X = X[X.SZA<80]
reg = LinearRegression().fit(X.GHI.values.reshape(-1,1), X.Delta.values)

X['GHIAdapted'] = X.GHI - reg.predict(X.GHI.values.reshape(-1,1))





plt.figure()
plt.plot(X.ghi, label="Meas")
plt.plot(X.GHI, label="Modelated")
plt.plot(X.GHIAdapted, label="Adapted")
plt.legend()
plt.show()  


m.rrmsd(X.ghi, X.GHI)
m.rrmsd(X.ghi, X.GHIAdapted)



data['Adapted']  = data.GHI - reg.predict(data.GHI.values.reshape(-1,1))

data = data[data.SZA<80]

inp = data[["latitude","longitude", "eth", "sza", "Adapted", "ghics"]].copy()

inp.columns = ['latitude', 'longitude', 'eth', 'sza', 'ghi', 'ghics']
pred3 = sm.get("Yang5").predict(inp)



Adaptado = GHI - (0.08345613*GHI -94.21355782982073)


s = data

s['pred1'] = pred.dni
s['pred2'] = pred2.dni
s['pred3'] = pred3.dni
t = s.dropna()


m.rrmsd(t.pred1, t.pred3)


plt.figure()
plt.plot(s.datetime, s.pred1, '.-b', label="Meas")
plt.plot(s.datetime, s.pred2, '.-r', label="CAMS")
plt.plot(s.datetime, s.pred3, '.-g', label="Adapted")
plt.legend()
plt.show()  

