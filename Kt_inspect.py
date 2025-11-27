import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Metrics as ms
from Geo import Geo
from Sites import Site
site_order = ["Yu", "Sa", "Sca", "Ero", "Lq"]

kt_bins = np.arange(0, 1.0, 0.05)
var_bins = np.linspace(0, 350, 30)

rrmsd_cams_all = {}
rrmsd_lsasaf_all = {}

# === Calcular RRMSD 2D por sitio ===
for site in site_order:
    s = Site(site.upper())
    df = pd.read_csv(f"{site.lower()}15.csv")
    df['datetime'] = pd.to_datetime(df.datetime)
    dfGeo = Geo(df.datetime, lat= s.lat, long=s.long, alt=s.alt, gmt=0, beta=0).df

    df["kt"] = df["ghi"] / dfGeo["GHIargp2"].values
    df["ghi_diff"] = df["ghi"].diff()
    df["var"] = df["ghi_diff"].rolling(window=10, center=True).std()

    rrmsd_grid_cams = np.full((len(kt_bins)-1, len(var_bins)-1), np.nan)
    rrmsd_grid_lsasaf = np.full_like(rrmsd_grid_cams, np.nan)

    for i in range(len(kt_bins) - 1):
        for j in range(len(var_bins) - 1):
            df_bin = df[
                (df["kt"] >= kt_bins[i]) & (df["kt"] < kt_bins[i+1]) &
                (df["var"] >= var_bins[j]) & (df["var"] < var_bins[j+1])
            ]
            if len(df_bin) > 10:
                rrmsd_grid_cams[i, j] = ms.rrmsd(df_bin["ghi"], df_bin["cams"])
                rrmsd_grid_lsasaf[i, j] = ms.rrmsd(df_bin["ghi"], df_bin["lsasaf"])

    rrmsd_cams_all[site] = rrmsd_grid_cams
    rrmsd_lsasaf_all[site] = rrmsd_grid_lsasaf

# === Determinar límites globales del colorbar ===
all_values = np.concatenate([
    np.ravel(list(rrmsd_cams_all.values())),
    np.ravel(list(rrmsd_lsasaf_all.values()))
])
valid_values = all_values[np.isfinite(all_values)]
vmin, vmax = np.nanpercentile(valid_values, [5, 95])

# === Crear figura combinada ===
fig, axes = plt.subplots(len(site_order), 2, figsize=(14, 10), sharex=True, sharey=True)

for i, site in enumerate(site_order):
    # Ahora X = var, Y = kt → transponemos las grillas y ajustamos extent
    im1 = axes[i, 0].imshow(rrmsd_cams_all[site],
                            origin="lower",
                            extent=[var_bins[0], var_bins[-1], kt_bins[0], kt_bins[-1]],
                            aspect="auto", cmap="Oranges", vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f"{site} - CAMS")

    im2 = axes[i, 1].imshow(rrmsd_lsasaf_all[site],
                            origin="lower",
                            extent=[var_bins[0], var_bins[-1], kt_bins[0], kt_bins[-1]],
                            aspect="auto", cmap="Oranges", vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f"{site} - LSA-SAF")

    if i == len(site_order) - 1:
        axes[i, 0].set_xlabel("Variabilidad (σ de ΔGHI)")
        axes[i, 1].set_xlabel("Variabilidad (σ de ΔGHI)")
    axes[i, 0].set_ylabel("kt")

# === Ajustar layout para colorbar y título ===
fig.subplots_adjust(right=0.88, top=0.93, hspace=0.3, wspace=0.15)

# === Colorbar global fuera del área de los plots ===
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label("rRMSD")

#plt.suptitle("RRMSD según variabilidad y kt para todos los sitios", fontsize=14)
plt.show()
