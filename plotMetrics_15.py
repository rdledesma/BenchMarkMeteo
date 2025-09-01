import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Definir sitios y modelos (solo CAMS y LSA-SAF)
sitios = ['Par', 'Flo','Bsa', 'Brb','Pil',  'Ptr', 'Yu',  'Sa', 'Sca', 'Er', 'Lq']
modelos = ['CAMS', 'LSA-SAF']

# --- Valores de las métricas para CAMS y LSA-SAF ---
MBE_values = [
    [4.0, 1.7, -0.5,-4.6,4.8, 7.5, 0.5, 3.5, 3.2, -23.7, -7.1],  # CAMS
    [19.6,10, 10.9, 8.1 ,10.5, 15.1, 10.7, 17.3, 11.9, -8.1, 3.7],  # LSA-SAF
]

MAE_values = [
    [27.8, 17.6,14.6 ,13.9, 19.8, 15.5, 18.4, 23.0, 23.7, 27.8, 16.2],
    [32.1, 20.7,16.4 , 13.7,21.7, 19.9, 19.0, 26.9, 22.9, 16.5, 12.3]
]

RMSE_values = [
    [40.7, 29.4, 23.1, 22,  31.6, 22.2, 28.4, 33.2, 31.3, 41.2, 25.3],
    [45.5, 33.1, 24.4, 22.4,34.2, 27.9, 28.8, 38.8, 31.4, 26.8, 22.1]
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
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = [('MBE', MBE_df), ('MAE', MAE_df), ('RMSE', RMSE_df)]

for ax, (metric_name, df_metric) in zip(axes, metrics):
    heatmap_data = df_metric.pivot(index='Modelo', columns='Sitio', values=metric_name)
    # Reordenar columnas según el orden deseado
    heatmap_data = heatmap_data.reindex(columns=sitios)
    
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={'label': metric_name})
    ax.set_title(f'{metric_name} por modelo y sitio')
    ax.set_xlabel("Sitio")
    ax.set_ylabel("Modelo 15 minutes")


plt.tight_layout()
plt.show()





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define sites and models (only CAMS and LSA-SAF)
sites = ['Par', 'Flo', 'Bsa', 'Brb', 'Pil', 'Ptr', 'Yu', 'Sa', 'Sca', 'Er', 'Lq']
models = ['CAMS', 'LSA-SAF']

# --- Metric values for CAMS and LSA-SAF ---
MBE_values = [
    [4.0, 1.7, -0.5, -4.6, 4.8, 7.5, 0.5, 3.5, 3.2, -23.7, -7.1],   # CAMS
    [19.6, 10, 10.9, 8.1, 10.5, 15.1, 10.7, 17.3, 11.9, -8.1, 3.7], # LSA-SAF
]

MAE_values = [
    [27.8, 17.6, 14.6, 13.9, 19.8, 15.5, 18.4, 23.0, 23.7, 27.8, 16.2],
    [32.1, 20.7, 16.4, 13.7, 21.7, 19.9, 19.0, 26.9, 22.9, 16.5, 12.3]
]

RMSE_values = [
    [40.7, 29.4, 23.1, 22, 31.6, 22.2, 28.4, 33.2, 31.3, 41.2, 25.3],
    [45.5, 33.1, 24.4, 22.4, 34.2, 27.9, 28.8, 38.8, 31.4, 26.8, 22.1]
]

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





for metric_name, df_metric in metrics.items():
    plt.figure(figsize=(12, 4))
    heatmap_data = df_metric.pivot(index="Model", columns="Site", values=metric_name)
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={'label': metric_name}, annot_kws={"size":12})
    plt.title(f"{metric_name} by model and site")
    plt.xlabel("Site")
    plt.ylabel("Model (15 minutes)")
    plt.tight_layout()
    plt.show()