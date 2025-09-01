import matplotlib.pyplot as plt
import pandas as pd
# Datos

lats = [-23.5844, -22.90477 ,-24.7288, -25.8951, -24.39278, -22.103936, -12.0500, -27.6047, -31.67, -34.60,-15.6010, 5.8060,-9.0690]
lons = [-64.5066, -64.662756,-65.4095, -65.925,  -65.76806, -65.599923, -75.3200, -48.5227, - 63.88, -58.48, -47.7130, -55.2146,-40.3200]
alts = [   401,  806  ,  1233,   1624,        3355 ,    3500,          3314, 11, 335, 30, 1023,4, 387]

sites = ['YU','ISCA' , 'SA','SCA', 'ERO','LQ','OHY', 'FLO','PIL', 'BSA','BRB','PAR','PTR','']


# Crear DataFrame
df = pd.DataFrame({
    "site": sites,
    "lat": lats,
    "lon": lons,
    "alt": alts
})

# Ordenar por altitud
df_sorted = df.sort_values(by="alt", ascending=True).reset_index(drop=True)
print(df_sorted)



# ======= Gráfico 3D =======
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(lons, lats, alts, c=alts, cmap='viridis', s=80)
for lon, lat, alt, name in zip(lons, lats, alts, sites):
    ax.text(lon, lat, alt, name, fontsize=8)
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_zlabel("Altitud (m)")
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Altitud (m)")
plt.title("Distribución de sitios en Lat, Lon y Alt (3D)")
plt.show()

# ======= Gráfico 2D (mapa lat/lon con color por altitud) =======
plt.figure(figsize=(8,6))
sc2 = plt.scatter(lons, lats, c=alts, cmap='viridis', s=100, edgecolor='k')
for lon, lat, name in zip(lons, lats, sites):
    plt.text(lon, lat, name, fontsize=8, ha='right')
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Mapa de sitios (color según altitud)")
cbar = plt.colorbar(sc2)
cbar.set_label("Altitud (m)")
plt.show()



import matplotlib.pyplot as plt

# ======= Perfil altitudinal (barras por sitio) =======
plt.figure(figsize=(10,5))
bars = plt.bar(df_sorted.site, df_sorted.alt, color='skyblue', edgecolor='k')

# Añadir texto sobre cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,   # posición x centrada
        height + 30,                       # un poco arriba de la barra
        f"{height}",                       # texto (la altitud)
        ha='center', va='bottom', fontsize=9
    )

plt.xticks(rotation=45)
plt.ylabel("Altitud (m)")
plt.title("Altitud de cada sitio")
plt.show()

# ======= Relación Latitud vs Altitud =======
plt.figure(figsize=(8,6))
plt.scatter(lats, alts, c=alts, cmap='plasma', s=100, edgecolor='k')
for lat, alt, name in zip(lats, alts, sites):
    plt.text(lat, alt, name, fontsize=8, ha='right')
plt.xlabel("Latitud")
plt.ylabel("Altitud (m)")
plt.title("Relación Latitud - Altitud")
cbar = plt.colorbar()
cbar.set_label("Altitud (m)")
plt.show()
