import pandas as pd
import matplotlib.pyplot as plt
import os
from Geo import Geo
from Sites import Site
import Metrics
import numpy as np
from datetime import date
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

cams = pd.read_csv(f'/home/inenco/Documentos/HELIOSAT/{str.lower(site.cod)}.csv', usecols=['GHI'], sep=";", 
                   header=42)

cams['datetime'] = pd.date_range(
    start="2013/01/01 00:00", end="2024/12/31 23:59", freq="15min"
)

cams['GHI'] = cams.GHI * 4

lsasaf = pd.read_csv(f'/home/inenco/Documentos/LSASAF/{str.lower(site.cod)}_15.csv')
lsasaf['datetime'] = pd.to_datetime(lsasaf.datetime)

lsasaf = (lsasaf.set_index('datetime')
      .reindex(pd.date_range(
    start="2013/01/01 00:00", end="2024/12/31 23:59", freq="15min"))
      .rename_axis(['datetime'])
      .reset_index())




from datetime import timedelta

dfGeo = Geo(range_dates=data.datetime + timedelta(minutes=7.5),
            lat=site.lat, long=site.long, alt=site.alt, gmt=0, beta=0).df

data['SZA'] = dfGeo.SZA.values
data['TOA'] = dfGeo.TOA.values
data['cams'] = cams.GHI.values
data['lsasaf'] = lsasaf.GHI.values



X = data.dropna()
X = X[X.SZA<80]

#X = X[X.ghi>1]
X.describe()


Metrics.rmbe(X.ghi, X.cams)
Metrics.rmbe(X.ghi, X.lsasaf)

Metrics.rmae(X.ghi, X.cams)
Metrics.rmae(X.ghi, X.lsasaf)


Metrics.rrmsd(X.ghi, X.cams)
Metrics.rrmsd(X.ghi, X.lsasaf)




X.to_csv('ero15.csv', index=False)

X['kt'] = data.ghi / data.TOA 

#X = X[X['kt']>0.1]

Metrics.rrmsd(X.ghi, X.cams)
Metrics.rrmsd(X.ghi, X.lsasaf)


plt.figure()
plt.plot(X.datetime, X.ghi, label="Meas")
plt.plot(X.datetime, X.cams, label="CAMS")
plt.plot(X.datetime, X.lsasaf, label="LSA-SAF")
plt.legend()
plt.show()




# Definir los límites de los bins de SZA
bins = np.arange(0, 90, 10)  # 0, 10, 20, ..., 80
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# Crear una columna con el bin correspondiente
X['SZA_bin'] = pd.cut(X['SZA'], bins=bins, labels=labels, right=False)

# Calcular el rrmsd por bin para cams y lsasaf
rrmsd_cams = X.groupby('SZA_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.cams))
rrmsd_lsasaf = X.groupby('SZA_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.lsasaf))

# Combinar resultados en un dataframe
rrmsd_df = pd.DataFrame({
    'RRMSD_cams': rrmsd_cams,
    'RRMSD_lsasaf': rrmsd_lsasaf
})

plt.figure()
rrmsd_df.plot(kind='bar')
plt.show(block=False)







# Definir los límites de los bins de SZA
bins = np.arange(0.1, 1, 0.1)  # 0, 10, 20, ..., 80
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# Crear una columna con el bin correspondiente
X['kt_bin'] = pd.cut(X['kt'], bins=bins, labels=labels, right=False)

# Calcular el rrmsd por bin para cams y lsasaf
rrmsd_cams = X.groupby('kt_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.cams))
rrmsd_lsasaf = X.groupby('kt_bin').apply(lambda g: Metrics.rrmsd(g.ghi, g.lsasaf))

# Combinar resultados en un dataframe
rrmsd_df = pd.DataFrame({
    'RRMSD_cams': rrmsd_cams,
    'RRMSD_lsasaf': rrmsd_lsasaf
})

plt.figure()
rrmsd_df.plot(kind='bar')
plt.show(block=False)



import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# GHI
plt.subplot(1, 3, 1)
plt.hist(X['ghi'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución GHI')
plt.xlabel('GHI')
plt.ylabel('Frecuencia')

# Cams
plt.subplot(1, 3, 2)
plt.hist(X['cams'], bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribución Cams')
plt.xlabel('Cams')

# Lsasaf
plt.subplot(1, 3, 3)
plt.hist(X['lsasaf'], bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribución Lsasaf')
plt.xlabel('Lsasaf')

plt.tight_layout()
plt.show(block=False)



import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

for col, color in zip(['ghi', 'cams', 'lsasaf'], ['blue', 'orange', 'green']):
    # Ordenar los valores
    sorted_vals = np.sort(X[col].dropna())
    # Calcular la CDF
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    # Graficar
    plt.plot(sorted_vals, cdf, label=col, color=color)

plt.xlabel('Valor')
plt.ylabel('CDF')
plt.title('Funciones de Distribución Acumulada')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()


"""
La Figura X muestra las funciones de distribución acumulada (CDF)
 de la irradiancia global horizontal (GHI) observada y estimada por los modelos CAMS y LSA-SAF.
   En la región de baja irradiancia (<100 W/m²),
   la curva de CAMS presenta una pendiente inicial más pronunciada, 
 indicando una mayor proporción de valores muy bajos en comparación con las observaciones. 
 Esto sugiere que el modelo tiende a sobreestimar la atenuación bajo condiciones de nubosidad o baja altura solar.
   Por el contrario, 
 LSA-SAF muestra un retraso en la acumulación en este rango, 
 lo que indica una menor frecuencia de valores bajos y, posiblemente,
   una sobreestimación sistemática de la irradiancia en estas condiciones. 
 En el rango intermedio (200–800 W/m²), las tres distribuciones son similares, aunque 
 las observaciones muestran una ligera mayor concentración de valores medios que ambos modelos. 
 En el extremo alto (>800 W/m²), las curvas convergen, 
 lo que indica que las simulaciones reproducen adecuadamente los valores máximos.
   En conjunto, los resultados sugieren 
 que las principales discrepancias entre modelos y observaciones se concentran en condiciones de baja irradiancia, 
 siendo más pronunciadas en CAMS que en LSA-SAF.

"""



import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

for col, color in zip(['ghi', 'cams', 'lsasaf'], ['blue', 'orange', 'green']):
    sorted_vals = np.sort(X[col].dropna())
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    plt.plot(sorted_vals, cdf, label=col, color=color)

plt.xscale('log')
plt.xlabel('Valor (W/m²) [escala log]')
plt.ylabel('CDF')
plt.title('Funciones de Distribución Acumulada (escala logarítmica)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.show()


"""

La Figura Y presenta las funciones de distribución acumulada (CDF) 
de la irradiancia global horizontal (GHI) observada y estimada por los modelos CAMS y LSA-SAF,
 utilizando escala logarítmica en el eje de irradiancia para resaltar el comportamiento en condiciones de baja radiación.
En el rango inferior a 50 W/m², 
CAMS muestra un incremento inicial más pronunciado que las observaciones,
lo que indica una mayor proporción de valores extremadamente bajos.
Esto sugiere que el modelo tiende a sobreestimar la atenuación bajo nubosidad densa o ángulos solares reducidos. En contraste,
 LSA-SAF retrasa el inicio de la acumulación, reflejando una subestimación sistemática de la frecuencia de valores bajos,
 posiblemente asociada a un suavizado o sobreestimación de la irradiancia en estas condiciones. Entre 50 y 200 W/m², 
las tres curvas convergen progresivamente, aunque CAMS sigue presentando mayor acumulación relativa que GHI, 
mientras que LSA-SAF lo hace de forma más lenta. En el rango intermedio (200–800 W/m²), 
las diferencias se reducen notablemente y las distribuciones de GHI y CAMS muestran mayor similitud que con LSA-SAF. 
Finalmente, para irradiancias superiores a 800 W/m², las tres series coinciden casi completamente,
 indicando que ambos modelos reproducen adecuadamente la distribución de valores máximos. Estos 
resultados confirman que las principales discrepancias entre modelos y observaciones se concentran 
en el dominio de baja irradiancia,  con sesgos de signo opuesto en CAMS y LSA-SAF.
"""





d1 = pd.read_csv('yu15.csv')
d2 = pd.read_csv('sa15.csv')
d3 = pd.read_csv('sca15.csv')
d4 = pd.read_csv('ero15.csv')
d5 = pd.read_csv('lq15.csv')

d2['datetime'] = pd.to_datetime(d2.datetime)
d2 = d2[d2.datetime.dt.year < 2019]


d1['site'] = 'yu'
d2['site'] = 'sa'
d3['site'] = 'sca'
d4['site'] = 'ero'
d5['site'] = 'lq'


X = pd.concat([d1,d2,d3, d4,d5])


import seaborn as sns




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- RRMSD vs SZA ---
bins_sza = np.arange(0, 90, 10)
labels_sza = [f"{bins_sza[i]}-{bins_sza[i+1]}" for i in range(len(bins_sza)-1)]
X['SZA_bin'] = pd.cut(X['SZA'], bins=bins_sza, labels=labels_sza, right=False)
# Calcular rrmsd por sitio y por bin
rrmsd_sza = (
    X.groupby(['site','SZA_bin'])
    .apply(lambda g: pd.Series({
        'RRMSD_cams': Metrics.rrmsd(g.ghi, g.cams),
        'RRMSD_lsasaf': Metrics.rrmsd(g.ghi, g.lsasaf)
    }))
    .reset_index()
)




# Plot multi-series (cada sitio = color distinto)
plt.figure(figsize=(10,5))
for site in rrmsd_sza['site'].unique():
    subset = rrmsd_sza[rrmsd_sza['site'] == site]
    plt.plot(subset['SZA_bin'], subset['RRMSD_cams'], marker='o', label=f"CAMS - {site}")
    plt.plot(subset['SZA_bin'], subset['RRMSD_lsasaf'], marker='s', linestyle='--', label=f"LSASAF - {site}")


# Etiquetas de ejes con fuente más grande
plt.xlabel("SZA bin", fontsize=20)
plt.ylabel("RRMSD", fontsize=20)

# Leyenda más grande
plt.legend(ncol=3, fontsize=16)
# Ticks más grandes
plt.tick_params(axis='both', labelsize=16)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

X['kt'] = X.ghi / X.TOA


# --- RRMSD vs Kt ---
bins_kt = np.arange(0.1, 1, 0.1)
labels_kt = [f"{bins_kt[i]:.1f}-{bins_kt[i+1]:.1f}" for i in range(len(bins_kt)-1)]
X['kt_bin'] = pd.cut(X['kt'], bins=bins_kt, labels=labels_kt, right=False)

rrmsd_kt = (
    X.groupby(['site','kt_bin'])
    .apply(lambda g: pd.Series({
        'RRMSD_cams': Metrics.rrmsd(g.ghi, g.cams),
        'RRMSD_lsasaf': Metrics.rrmsd(g.ghi, g.lsasaf)
    }))
    .reset_index()
)

plt.figure(figsize=(10,5))

for site in rrmsd_kt['site'].unique():
    subset = rrmsd_kt[rrmsd_kt['site'] == site]
    plt.plot(subset['kt_bin'], subset['RRMSD_cams'], marker='o', label=f"CAMS - {site}")
    plt.plot(subset['kt_bin'], subset['RRMSD_lsasaf'], marker='s', linestyle='--', label=f"LSASAF - {site}")

# Etiquetas de ejes con fuente más grande
plt.xlabel("Kt bin", fontsize=20)
plt.ylabel("RRMSD", fontsize=20)

# Leyenda más grande
plt.legend(ncol=3, fontsize=16)

# Ticks más grandes
plt.tick_params(axis='both', labelsize=16)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()






import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(X['ghi'], bins=30, kde=True, color="skyblue", edgecolor="black")
plt.xlabel("GHI")
plt.ylabel("Frecuencia")
plt.title("Distribución de GHI con densidad")
plt.show()



plt.figure(figsize=(6,6))
plt.hexbin(X['ghi'], X['cams'], gridsize=50, cmap='viridis')
plt.plot([0, X['ghi'].max()], [0, X['ghi'].max()], 'r--', label="1:1 line")
plt.colorbar(label="Densidad de puntos")
plt.xlabel("Observado (GHI)")
plt.ylabel("Estimado (CAMS)")
plt.legend()
plt.title("Dispersión GHI vs CAMS con densidad")
plt.show()



X['error_cams'] = X['cams'] - X['ghi']
plt.figure(figsize=(10,5))
sns.boxplot(x=X['SZA_bin'], y=X['error_cams'])
plt.xlabel("SZA bin")
plt.ylabel("Error CAMS (W/m²)")
plt.title("Distribución de errores por ángulo cenital solar")
plt.xticks(rotation=45)
plt.show()




plt.figure(figsize=(8,5))
sns.ecdfplot(X['ghi'], label="Observado (GHI)")
sns.ecdfplot(X['cams'], label="CAMS")
sns.ecdfplot(X['lsasaf'], label="LSASAF")
plt.xlabel("Valor")
plt.ylabel("Probabilidad acumulada")
plt.legend()
plt.title("Funciones de distribución acumulada")
plt.show()





import seaborn as sns

for site in X['site'].unique():
    subset = X[X['site'] == site]
    plt.figure(figsize=(8,5))
    sns.histplot(subset['ghi'], bins=30, kde=True, color="black", label="GHI", stat="density")
    sns.histplot(subset['cams'], bins=30, kde=True, color="orange", label="CAMS", stat="density", alpha=0.5)
    sns.histplot(subset['lsasaf'], bins=30, kde=True, color="green", label="LSASAF", stat="density", alpha=0.5)
    plt.xlabel("Irradiancia (W/m²)")
    plt.ylabel("Densidad")
    plt.title(f"Distribución de irradiancia - {site}")
    plt.legend()
    plt.tight_layout()
    plt.show()




for site in X['site'].unique():
    subset = X[X['site'] == site]

    plt.figure(figsize=(12,5))

    # CAMS
    plt.subplot(1,2,1)
    plt.hexbin(subset['ghi'], subset['cams'], gridsize=50, cmap="viridis")
    plt.plot([0, subset['ghi'].max()], [0, subset['ghi'].max()], 'r--')
    plt.colorbar(label="Densidad")
    plt.xlabel("Observado (GHI)")
    plt.ylabel("CAMS")
    plt.title(f"Scatter GHI vs CAMS - {site}")

    # LSASAF
    plt.subplot(1,2,2)
    plt.hexbin(subset['ghi'], subset['lsasaf'], gridsize=50, cmap="viridis")
    plt.plot([0, subset['ghi'].max()], [0, subset['ghi'].max()], 'r--')
    plt.colorbar(label="Densidad")
    plt.xlabel("Observado (GHI)")
    plt.ylabel("LSASAF")
    plt.title(f"Scatter GHI vs LSASAF - {site}")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

for site in X['site'].unique():
    subset = X[X['site'] == site]

    plt.figure(figsize=(12,5))

    # CAMS
    plt.subplot(1,2,1)
    hb = plt.hexbin(subset['ghi'], subset['cams'], gridsize=100, cmap="viridis", mincnt=1)
    plt.plot([0, subset['ghi'].max()], [0, subset['ghi'].max()], 'r--', lw=2, label="1:1")
    cb = plt.colorbar(hb)
    cb.set_label("Número de puntos", fontsize=14)
    plt.xlabel("Observado (GHI)", fontsize=16)
    plt.ylabel("Estimado (CAMS)", fontsize=16)
    plt.title(f"{site} - GHI vs CAMS", fontsize=18)
    plt.axis("equal")
    plt.xlim(100, subset['ghi'].max())
    plt.ylim(0, subset['ghi'].max())
    plt.legend(fontsize=12)

    # LSASAF
    plt.subplot(1,2,2)
    hb = plt.hexbin(subset['ghi'], subset['lsasaf'], gridsize=100, cmap="viridis", mincnt=1)
    plt.plot([0, subset['ghi'].max()], [0, subset['ghi'].max()], 'r--', lw=2, label="1:1")
    cb = plt.colorbar(hb)
    cb.set_label("Número de puntos", fontsize=14)
    plt.xlabel("Observado (GHI)", fontsize=16)
    plt.ylabel("Estimado (LSASAF)", fontsize=16)
    plt.title(f"{site} - GHI vs LSASAF", fontsize=18)
    plt.axis("equal")
    plt.xlim(0, subset['ghi'].max())
    plt.ylim(0, subset['ghi'].max())
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # ajustar espacio entre subplots
    plt.show()
