import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Definir sitios y modelos
sitios = ['Par', 'Flo','Bsa', 'Pil', 'Ptr','Yu', 'Brb', 'Sa', 'Sca', 'Ero','Lq']
modelos  = ['CAMS', 'LSA-SAF', 'ERA-5', 'MERRA-2']

# --- Valores de las métricas: modificar aquí fácilmente ---
MBE_values = [
    [4,    1.9  , -0.2, -4.3, 7.8,   0.5, 5.0 ,  3.6 , 2.9, -23,7, -6.1 ],  # CAMS
    [19.7, 10.1 , 11.2,  8.5, 15.4, 10.7, 10.3, 17.3, 11.2, -8.1 ,  4.6],  # LSA-SAF
    [2.1, -0.1  , 5.6,  -0.5, 0,    0.4 , 1.1 , 8.5 ,  8.7, -13.7, -1.7],  # ERA-5
    [12.4, 17.7 , 11.9,  0.1, 17,   26.9, 7.0 , 42.1, 13.1, -3.9 ,  1.0]   # MERRA-2
]

MAE_values = [
    [22  , 22.2, 19.2, 17.7, 19.4, 16.1, 16.3, 20.5 , 21.4, 26.8, 14.6],
    [27.7, 24.0, 20.5, 17.6, 23.0, 21.7, 18.6, 24.8 , 20.5, 15.0, 10.9],
    [35.2, 30.8, 24.4, 22.4, 25.2, 22.7, 21.3, 26.8 , 21.4, 19.1, 12.0],
    [34.1, 37.7, 28.3, 34.6, 28.9, 35.0, 23.5, 47.0 , 22.0, 13.1, 13.4]
]

RMSE_values = [
    [32.0, 32.3, 26.6, 25.3, 25.2, 24.1, 25.7, 28.8 , 27.3, 39.5, 22.0],
    [37.8, 34.3, 27.2, 24.2, 30.3, 29.2, 28.4, 35   , 27.1, 24.3, 18.7],
    [46.8, 42  , 33.5, 31.5, 32.9, 34.6, 32.1, 37.5 , 28.9, 25.3, 19.3],
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
    for model, row in zip(modelos, values):
        for site, val in zip(sitios, row):
            data.append({'Modelo': model, 'Sitio': site, metric_name: val})
    return pd.DataFrame(data)

MBE_df = create_metric_df(MBE_values, 'rMBE %')
MAE_df = create_metric_df(MAE_values, 'rMAE %')
RMSE_df = create_metric_df(RMSE_values, 'rRMSE %')

# --- Plot heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = [('rMBE %', MBE_df), ('rMAE %', MAE_df), ('rRMSE %', RMSE_df)]

for ax, (metric_name, df_metric) in zip(axes, metrics):
    heatmap_data = df_metric.pivot(index='Modelo', columns='Sitio', values=metric_name)
    # Reorder columns according to the desired site order
    heatmap_data = heatmap_data.reindex(columns=sitios)
    
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

sites = ['Yu', 'Sa', 'Sca', 'Ero', 'Lq']

# Crear figura con un subplot por cada métrica
fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 12), sharex=True)

# Iterar sobre cada métrica y su DataFrame
for ax, (metric_name, df_metric) in zip(axes, metrics.items()):
    # Filtrar solo los sitios deseados
    df_metric = df_metric[df_metric["Sitio"].isin(sites)]
    
    # Hacer el barplot
    sns.barplot(data=df_metric, x="Sitio", y=metric_name, hue="Modelo", ax=ax)
    
    # Etiquetas
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel("")

# Etiqueta en el último eje
axes[-1].set_xlabel("Sitio", fontsize=16)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax, (metric_name, df_metric) in zip(axes, metrics.items()):
    sns.barplot(data=df_metric, x="Sitio", y=metric_name, hue="Modelo", ax=ax)
    #ax.set_title(f"{metric_name} by site", fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel("")

axes[-1].set_xlabel("Site", fontsize=16)
plt.tight_layout()
plt.show()










import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

for ax, (metric_name, df_metric) in zip(axes, metrics.items()):
    sns.barplot(data=df_metric, x="Sitio", y=metric_name, hue="Modelo", ax=ax, palette="Greys")
    #ax.set_title(f"{metric_name} by site", fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel("")

axes[-1].set_xlabel("Site", fontsize=16)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

# Lista para guardar handles y labels de la leyenda
handles, labels = [], []

# Crear los gráficos
for i, (ax, (metric_name, df_metric)) in enumerate(zip(axes, metrics.items())):
    # Al primer gráfico le permitimos generar la leyenda
    if i == 0:
        sns.barplot(data=df_metric, x="Sitio", y=metric_name, hue="Modelo", ax=ax, palette="Greys")
        handles, labels = ax.get_legend_handles_labels()  # Captura los handles y labels
    else:
        sns.barplot(data=df_metric, x="Sitio", y=metric_name, hue="Modelo", ax=ax, palette="Greys", legend=False)
    
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel("")

# Colocar la leyenda en la parte superior, en horizontal
# Aseguramos que 'ncol' sea suficientemente grande para acomodar todos los elementos en una fila
fig.legend(handles, labels, loc='upper center', ncol=10, bbox_to_anchor=(0.5, 1.05), fontsize=16, frameon=False)

# Ajustar el texto del eje X en el último gráfico
axes[-1].set_xlabel("Site", fontsize=16)

# Ajustar la distribución del layout para que no se solapen
plt.tight_layout()
plt.show()






import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

# Lista para guardar handles y labels de la leyenda
handles, labels = [], []

# Crear los gráficos
for i, (ax, (metric_name, df_metric)) in enumerate(zip(axes, metrics.items())):
    # Al primer gráfico le permitimos generar la leyenda
    if i == 0:
        sns.barplot(data=df_metric, x="Sitio", y=metric_name, hue="Modelo", ax=ax, palette="Greys")
        handles, labels = ax.get_legend_handles_labels()  # Captura los handles y labels
    else:
        sns.barplot(data=df_metric, x="Sitio", y=metric_name, hue="Modelo", ax=ax, palette="Greys", legend=False)
    
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel("")

# Ajustar la leyenda para que sea horizontal
fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05), fontsize=16, frameon=False)

# Ajustar el espacio entre los subgráficos y la leyenda para evitar solapamiento
fig.subplots_adjust(hspace=0.3)  # Ajustar el espacio vertical entre los subgráficos y la leyenda

# Ajustar el texto del eje X en el último gráfico
axes[-1].set_xlabel("Site", fontsize=16)

# Mostrar el gráfico
plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

handles, labels = [], []

for i, (ax, (metric_name, df_metric)) in enumerate(zip(axes, metrics.items())):
    if i == 0:
        sns.barplot(
            data=df_metric,
            x="Sitio",
            y=metric_name,
            hue="Modelo",
            ax=ax,
            palette="Greys"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()
    else:
        sns.barplot(
            data=df_metric,
            x="Sitio",
            y=metric_name,
            hue="Modelo",
            ax=ax,
            palette="Greys",
            legend=False
        )

    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel("")

    # ✅ Límites y ticks explícitos para RMAE y RMSD
    if metric_name in ["rMAE %", "rRMSE %"]:
        ax.set_ylim(0, 67)
        ax.set_yticks(np.arange(0, 71, 10))

# Leyenda horizontal
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    fontsize=16,
    frameon=False
)

axes[-1].set_xlabel("Site", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
