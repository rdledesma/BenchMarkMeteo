import pandas as pd
import matplotlib.pyplot as plt
import os
from Geo import Geo
from Sites import Site
import Metrics
import numpy as np
from datetime import date
site = Site('PTR')

data = pd.read_csv(f'/home/inenco/Documentos/Medidas/{str.lower(site.cod)}_15.csv')
data['datetime'] = pd.to_datetime(data.datetime)


data = (data.set_index('datetime')
      .reindex(pd.date_range(
    start="2010/01/01 00:00", end="2019/12/31 23:59", freq="15min"))
      .rename_axis(['datetime'])
      .reset_index())



plt.figure()
plt.plot(data.datetime, data.ghi, label="Meas")
plt.show(block=False)





cams = pd.read_csv(f'/home/inenco/Documentos/HELIOSAT/{str.lower(site.cod)}.csv', usecols=['GHI'], sep=";", 
                   header=42)

cams['datetime'] = pd.date_range(
    start="2010/01/01 00:00", end="2019/12/31 23:59", freq="15min"
)

cams['GHI'] = cams.GHI * 4
cams = cams.resample(on='datetime', rule='60 min').mean().reset_index()



lsasaf = pd.read_csv(f'/home/inenco/Documentos/LSASAF/{str.lower(site.cod)}_15.csv')
lsasaf['datetime'] = pd.to_datetime(lsasaf.datetime)

lsasaf = (lsasaf.set_index('datetime')
      .reindex(pd.date_range(
    start="2010/01/01 00:00", end="2019/12/31 23:59", freq="15min"))
      .rename_axis(['datetime'])
      .reset_index())

lsasaf = lsasaf.resample(on='datetime', rule='60 min').mean().reset_index()






era =  pd.read_csv(f'/home/inenco/Documentos/ERA/{site.cod}_era.csv')
era['datetime'] = pd.to_datetime(era.datetime)


merra =  pd.read_csv(f'/home/inenco/Documentos/MERRA/{site.cod}.csv')
merra['datetime'] = pd.to_datetime(merra.datetime)
merra = (merra.set_index('datetime')
      .reindex(pd.date_range(
    start="2010/01/01 00:00", end="2019/12/31 23:59", freq="60min"))
      .rename_axis(['datetime'])
      .reset_index())


plt.figure()
plt.plot(lsasaf.GHI)
plt.show(block=False)


from datetime import timedelta

dfGeo = Geo(range_dates=data.datetime + timedelta(minutes=30),
            lat=site.lat, long=site.long, alt=site.alt, gmt=0, beta=0).df

data['SZA'] = dfGeo.SZA.values
data['TOA'] = dfGeo.TOA.values
data['cams'] = cams.GHI.values
data['lsasaf'] = lsasaf.GHI.values
data['era'] = era.GHI.values
data['merra'] = merra.GHI.values








X = data.dropna()
X = X[X.SZA<80]
X.describe()


X.to_csv('ptr60.csv', index=False)

X['kt'] = data.ghi / data.TOA

#X = X[X['kt']>0.1]


# Metrics.rmbe(X.ghi, X.cams)
# Metrics.rmbe(X.ghi, X.lsasaf)

Metrics.rmbe(X.ghi, X.cams)
Metrics.rmbe(X.ghi, X.lsasaf)
Metrics.rmbe(X.ghi, X.era)
Metrics.rmbe(X.ghi, X.merra)


# plt.figure()
# plt.plot(X.datetime, X.ghi, label="Meas")
# plt.plot(X.datetime, X.cams, label="CAMS")
# plt.plot(X.datetime, X.lsasaf, label="LSA-SAF")
# plt.legend()
# plt.show(block=False)




# # Definir los límites de los bins de SZA
# bins = np.arange(0, 90, 10)  # 0, 10, 20, ..., 80
# labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# # Crear una columna con el bin correspondiente
# X['SZA_bin'] = pd.cut(X['SZA'], bins=bins, labels=labels, right=False)

# # Calcular el rrmsd por bin para cams y lsasaf
# rrmsd_cams = X.groupby('SZA_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.cams))
# rrmsd_lsasaf = X.groupby('SZA_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.lsasaf))

# # Combinar resultados en un dataframe
# rrmsd_df = pd.DataFrame({
#     'RRMSD_cams': rrmsd_cams,
#     'RRMSD_lsasaf': rrmsd_lsasaf
# })

# plt.figure()
# rrmsd_df.plot(kind='bar')
# plt.show(block=False)







# # Definir los límites de los bins de SZA
# bins = np.arange(0.1, 1, 0.1)  # 0, 10, 20, ..., 80
# labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# # Crear una columna con el bin correspondiente
# X['kt_bin'] = pd.cut(X['kt'], bins=bins, labels=labels, right=False)

# # Calcular el rrmsd por bin para cams y lsasaf
# rrmsd_cams = X.groupby('kt_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.cams))
# rrmsd_lsasaf = X.groupby('kt_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.lsasaf))

# # Combinar resultados en un dataframe
# rrmsd_df = pd.DataFrame({
#     'RRMSD_cams': rrmsd_cams,
#     'RRMSD_lsasaf': rrmsd_lsasaf
# })

# plt.figure()
# rrmsd_df.plot(kind='bar')
# plt.show(block=False)



# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 4))

# # GHI
# plt.subplot(1, 3, 1)
# plt.hist(X['ghi'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
# plt.title('Distribución GHI')
# plt.xlabel('GHI')
# plt.ylabel('Frecuencia')

# # Cams
# plt.subplot(1, 3, 2)
# plt.hist(X['cams'], bins=30, alpha=0.7, color='orange', edgecolor='black')
# plt.title('Distribución Cams')
# plt.xlabel('Cams')

# # Lsasaf
# plt.subplot(1, 3, 3)
# plt.hist(X['lsasaf'], bins=30, alpha=0.7, color='green', edgecolor='black')
# plt.title('Distribución Lsasaf')
# plt.xlabel('Lsasaf')

# plt.tight_layout()
# plt.show(block=False)



# import numpy as np
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))

# for col, color in zip(['ghi', 'cams', 'lsasaf'], ['blue', 'orange', 'green']):
#     # Ordenar los valores
#     sorted_vals = np.sort(X[col].dropna())
#     # Calcular la CDF
#     cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
#     # Graficar
#     plt.plot(sorted_vals, cdf, label=col, color=color)

# plt.xlabel('Valor')
# plt.ylabel('CDF')
# plt.title('Funciones de Distribución Acumulada')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.show(block=False)





# """
# La Figura X muestra las funciones de distribución acumulada (CDF)
#  de la irradiancia global horizontal (GHI) observada y estimada por los modelos CAMS y LSA-SAF.
#    En la región de baja irradiancia (<100 W/m²),
#    la curva de CAMS presenta una pendiente inicial más pronunciada, 
#  indicando una mayor proporción de valores muy bajos en comparación con las observaciones. 
#  Esto sugiere que el modelo tiende a sobreestimar la atenuación bajo condiciones de nubosidad o baja altura solar.
#    Por el contrario, 
#  LSA-SAF muestra un retraso en la acumulación en este rango, 
#  lo que indica una menor frecuencia de valores bajos y, posiblemente,
#    una sobreestimación sistemática de la irradiancia en estas condiciones. 
#  En el rango intermedio (200–800 W/m²), las tres distribuciones son similares, aunque 
#  las observaciones muestran una ligera mayor concentración de valores medios que ambos modelos. 
#  En el extremo alto (>800 W/m²), las curvas convergen, 
#  lo que indica que las simulaciones reproducen adecuadamente los valores máximos.
#    En conjunto, los resultados sugieren 
#  que las principales discrepancias entre modelos y observaciones se concentran en condiciones de baja irradiancia, 
#  siendo más pronunciadas en CAMS que en LSA-SAF.

# """



# import numpy as np
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))

# for col, color in zip(['ghi', 'cams', 'lsasaf'], ['blue', 'orange', 'green']):
#     sorted_vals = np.sort(X[col].dropna())
#     cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
#     plt.plot(sorted_vals, cdf, label=col, color=color)

# plt.xscale('log')
# plt.xlabel('Valor (W/m²) [escala log]')
# plt.ylabel('CDF')
# plt.title('Funciones de Distribución Acumulada (escala logarítmica)')
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.legend()
# plt.show()


# """

# La Figura Y presenta las funciones de distribución acumulada (CDF) 
# de la irradiancia global horizontal (GHI) observada y estimada por los modelos CAMS y LSA-SAF,
#  utilizando escala logarítmica en el eje de irradiancia para resaltar el comportamiento en condiciones de baja radiación.
# En el rango inferior a 50 W/m², 
# CAMS muestra un incremento inicial más pronunciado que las observaciones,
# lo que indica una mayor proporción de valores extremadamente bajos.
# Esto sugiere que el modelo tiende a sobreestimar la atenuación bajo nubosidad densa o ángulos solares reducidos. En contraste,
#  LSA-SAF retrasa el inicio de la acumulación, reflejando una subestimación sistemática de la frecuencia de valores bajos,
#  posiblemente asociada a un suavizado o sobreestimación de la irradiancia en estas condiciones. Entre 50 y 200 W/m², 
# las tres curvas convergen progresivamente, aunque CAMS sigue presentando mayor acumulación relativa que GHI, 
# mientras que LSA-SAF lo hace de forma más lenta. En el rango intermedio (200–800 W/m²), 
# las diferencias se reducen notablemente y las distribuciones de GHI y CAMS muestran mayor similitud que con LSA-SAF. 
# Finalmente, para irradiancias superiores a 800 W/m², las tres series coinciden casi completamente,
#  indicando que ambos modelos reproducen adecuadamente la distribución de valores máximos. Estos 
# resultados confirman que las principales discrepancias entre modelos y observaciones se concentran 
# en el dominio de baja irradiancia,  con sesgos de signo opuesto en CAMS y LSA-SAF.
# """