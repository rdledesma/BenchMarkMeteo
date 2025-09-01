import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Definir sitios y modelos
sites = ['Par', 'Flo','Bsa', 'Pil', 'Ptr','Yu', 'Brb', 'Sa', 'Sca', 'Ero','Lq']
models  = ['CAMS', 'LSA-SAF', 'ERA-5', 'MERRA-2']

# --- Valores de las métricas: modificar aquí fácilmente ---
MBE_values = [
    [4,    1.9  , -0.2, -4.3, 7.8,   0.5, 5.0 ,  3.6 , 2.9, -23,7, -6.1 ],  # CAMS
    [19.7, 10.1 , 11.2,  8.5, 15.4, 10.7, 10.3, 17.3, 11.2, -8.1 ,  4.6],  # LSA-SAF
    [2.1, -0.1  , 5.6,  -0.5, 0,   -4.2 , 1.1 , 8.5 ,  8.7, -13.7, -1.7],  # ERA-5
    [12.4, 17.7 , 11.9,  0.1, 17,   26.9, 7.0 , 42.1, 13.1, -3.9 ,  1.0]   # MERRA-2
]

MAE_values = [
    [22  , 22.2, 19.2, 17.7, 19.4, 16.1, 16.3, 20.5 , 21.4, 26.8, 14.6],
    [27.7, 24.0, 20.5, 17.6, 23.0, 21.7, 18.6, 24.8 , 20.5, 15.0, 10.9],
    [35.2, 30.8, 24.4, 22.4, 25.2, 45.4, 21.3, 26.8 , 21.4, 19.1, 12.0],
    [34.1, 37.7, 28.3, 34.6, 28.9, 35.0, 23.5, 47.0 , 22.0, 13.1, 13.4]
]

RMSE_values = [
    [32.0, 32.3, 26.6, 25.3, 25.2, 24.1, 25.7, 28.8 , 27.3, 39.5, 22.0],
    [37.8, 34.3, 27.2, 24.2, 30.3, 29.2, 28.4, 35   , 27.1, 24.3, 18.7],
    [46.8, 42  , 33.5, 31.5, 32.9, 61.9, 32.1, 37.5 , 28.9, 25.3, 19.3],
    [45.3, 52.2, 40.4, 45.3, 39.0, 51.9, 35.0, 63.6 , 29.3, 20.5, 21.1]
]

# --- Convertir a DataFrame ---
def create_metric_df(values, metric_name):
    data = []
    for modelo, row in zip(modelos, values):
        for sitio, val in zip(sitios, row):
            data.append({'Modelo': modelo, 'Sitio': sitio, metric_name: val})
    return pd.DataFrame(data)

MBE_df = create_metric_df(MBE_values, 'MBE')
MAE_df = create_metric_df(MAE_values, 'MAE')
RMSE_df = create_metric_df(RMSE_values, 'RMSE')

# --- Graficar heatmaps ---
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics = [('MBE', MBE_df), ('MAE', MAE_df), ('RMSE', RMSE_df)]


for ax, (metric_name, df_metric) in zip(axes, metrics):
    heatmap_data = df_metric.pivot(index='Modelo', columns='Sitio', values=metric_name)
    
    # Reordenar filas (modelos) y columnas (sitios)
    heatmap_data = heatmap_data.reindex(index=modelos, columns=sitios)
    
    sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
        ax=ax, cbar_kws={'label': metric_name}
    )
    ax.set_title(f'{metric_name} por modelo y sitio')
    ax.set_xlabel("Sitio")
    ax.set_ylabel("Modelo Horario")

plt.tight_layout()
plt.show()











import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Convert lists to DataFrames ---
def create_metric_df(values, metric_name):
    data = []
    for model, row in zip(models, values):
        for site, val in zip(sites, row):
            data.append({'Model': model, 'Site': site, metric_name: val})
    return pd.DataFrame(data)

MBE_df = create_metric_df(MBE_values, 'rMBE %')
MAE_df = create_metric_df(MAE_values, 'rMAE %')
RMSE_df = create_metric_df(RMSE_values, 'rRMSE %')

# --- Plot heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = [('rMBE %', MBE_df), ('rMAE %', MAE_df), ('rRMSE %', RMSE_df)]

for ax, (metric_name, df_metric) in zip(axes, metrics):
    heatmap_data = df_metric.pivot(index='Model', columns='Site', values=metric_name)
    # Reorder columns according to the desired site order
    heatmap_data = heatmap_data.reindex(columns=sites)
    
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, 
                cbar_kws={'label': metric_name})
    ax.set_title(f'{metric_name} by model and site')
    ax.set_xlabel("Site")
    ax.set_ylabel("Model (15 minutes)")

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Increase global font size
plt.rcParams.update({
    "font.size": 16,          # base font size
    "axes.titlesize": 18,     # title size
    "axes.labelsize": 16,     # x and y labels
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 14,    # y tick labels
    "legend.fontsize": 14,    # legend
    "figure.titlesize": 20    # figure title
})

metrics = {
    "rMBE %": MBE_df,
    "rMAE %": MAE_df,
    "rRMSE %": RMSE_df
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax, (metric_name, df_metric) in zip(axes, metrics.items()):
    sns.barplot(data=df_metric, x="Site", y=metric_name, hue="Model", ax=ax)
    #ax.set_title(f"{metric_name} by site", fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel("")

axes[-1].set_xlabel("Site", fontsize=16)
plt.tight_layout()
plt.show()

