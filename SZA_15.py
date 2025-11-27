import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# ✅ Contexto adecuado para paper
sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")

# Site order by altitude
site_order = [
    "Par", "Flo", "Bsa", "Pil", "Ptr", "Yu", 
    "Brb", "Sa", "Sca", "Er", "Lq"
]

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

path = "SZA-15/*.csv"
dfs = []

for file in glob.glob(path):
    site_raw = os.path.basename(file).split("_")[0].upper()
    site = site_map.get(site_raw, site_raw)

    df = pd.read_csv(file)

    df_long = df.melt(
        id_vars="SZA_bin",
        value_vars=["RRMSD_cams", "RRMSD_lsasaf"],
        var_name="Model",
        value_name="rRMSE"
    )

    df_long["Model"] = (
        df_long["Model"]
        .str.replace("RRMSD_", "", regex=False)
        .str.upper()
    )
    df_long["Site"] = site

    dfs.append(df_long)

df_all = pd.concat(dfs, ignore_index=True)
df_all["SZA_bin_end"] = df_all["SZA_bin"].str.split("-").str[1].astype(float)

# ✅ Marcadores claramente distinguibles (clave en B/N)
markers = ["o", "s", "^", "D", "X", "P"]

# ✅ FacetGrid
g = sns.FacetGrid(
    df_all,
    col="Site",
    col_wrap=3,
    col_order=site_order,
    height=4,
    sharey=True
)

# ✅ Líneas negras + marcadores distintos
g.map_dataframe(
    sns.lineplot,
    x="SZA_bin_end",
    y="rRMSE",
    hue="Model",
    style="Model",
    markers=markers,
    dashes=False,
    color="black",
    linewidth=1,
    markersize=12,
    markeredgewidth=1.5,
    markeredgecolor="black"
)

g.set_axis_labels("SZA bin end (°)", "rRMSE")
g.set(ylim=(0, 90))

# ✅ Leyenda única, horizontal y clara
g.add_legend(
    loc="upper center",
    ncols=4,
    frameon=False
)

plt.subplots_adjust(
    top=0.88,
    bottom=0.15,
    left=0.07,
    right=0.95,
    hspace=0.3,
    wspace=0.05
)

plt.show()
