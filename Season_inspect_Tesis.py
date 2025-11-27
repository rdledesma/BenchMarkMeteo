import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# Aumentar el tamaño de fuente general para todos los elementos
sns.set_context("talk", font_scale=1.2)  # "talk" es bueno para artículos, "poster" aún más grande

# Orden de los sitios según altitud (de menor a mayor)
site_order = [
    "Par", "Flo", "Bsa", "Pil", "Ptr", "Yu", 
    "Brb", "Sa", "Sca", "Er", "Lq"
]

# Diccionario de mapeo: códigos de archivo a nombres estandarizados de sitios
site_map = {
    "PAR": "Par",
    "FLO": "Flo",
    "BSA": "Bsa",
    "PIL": "Pil",
    "PTR": "Ptr",
    "YU":  "Yu",
    "BRB": "Brb",
    "SA":  "Sa",
    "SCA": "Sca",
    "ERO": "Er",
    "LQ":  "Lq"
}

path = "Season/*.csv"
dfs = []

for file in glob.glob(path):
    site_raw = os.path.basename(file).split("_")[0].upper()
    site = site_map.get(site_raw, site_raw)  # estandarizar nombre del sitio
    
    df = pd.read_csv(file)
    
    # Convertir a formato largo
    df_long = df.melt(
        id_vars="season",
        value_vars=["cams", "lsasaf"],
        var_name="Modelo",
        value_name="rRMSE"
    )
    
    df_long["Modelo"] = df_long["Modelo"].str.replace("RRMSD_", "", regex=False).str.upper()
    df_long["Sitio"] = site
    
    dfs.append(df_long)

df_all = pd.concat(dfs, ignore_index=True)


# Diccionario para traducir estaciones
season_map = {
    "Spring": "Primavera",
    "Summer": "Verano",
    "Autumn": "Otoño",
    "Winter": "Invierno"
}

# Reemplazar la columna 'season' con los nombres en español
df_all["season"] = df_all["season"].map(season_map)






#df_all["season"] = df_all["season"].str.split("-").str[1].astype(float)

# Filtrar sitios seleccionados
df_all = df_all[df_all.Sitio.isin(['Yu','Sa','Sca','Er','Lq'])]
site_order = ['Yu','Sa','Sca','Er','Lq']

# FacetGrid ordenado por altitud
g = sns.FacetGrid(
    df_all,
    col="Sitio",
    col_wrap=3,
    col_order=site_order,
    height=4,    # subgráficos ligeramente más grandes
    sharey=True
)

#g.map_dataframe(sns.lineplot, x="SZA_bin", y="rRMSE", hue="Modelo", marker="o")
g.map_dataframe(sns.lineplot, x="season", y="rRMSE", hue="Modelo", marker="o")
g.set_axis_labels("Temporada", "rRMSE")
#g.set(ylim=(0, 90))
g.add_legend(loc="upper center", ncols=4 )
#g.set_axis_labels("Bin de SZA (°)", "rRMSE")
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.05, right=0.95, hspace=0.3, wspace=0.15)
#plt.subplots_adjust(top=0.9)
#g.fig.suptitle("Comparación de rRMSE vs SZA (ordenado por altitud)", fontsize=16)

plt.show()
