import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# Increase general font size for all elements
sns.set_context("talk", font_scale=1.2)  # "talk" es bueno para artículos, "poster" aún más grande

# Site order by altitude (from lowest to highest)
site_order = [
    "Par", "Flo", "Bsa", "Pil", "Ptr", "Yu", 
    "Brb", "Sa", "Sca", "Er", "Lq"
]

# Mapping dictionary: file codes to standardized site names
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

path = "SZA-60/*.csv"
dfs = []

for file in glob.glob(path):
    site_raw = os.path.basename(file).split("_")[0].upper()
    site = site_map.get(site_raw, site_raw)  # standardize site name
    
    df = pd.read_csv(file)
    
    # Convert to long format
    df_long = df.melt(
        id_vars="SZA_bin",
        value_vars=["RRMSD_cams", "RRMSD_lsasaf", "RRMSD_era", "RRMSD_merra"],
        var_name="Model",
        value_name="rRMSE"
    )
    
    df_long["Model"] = df_long["Model"].str.replace("RRMSD_", "", regex=False).str.upper()
    df_long["Site"] = site
    
    dfs.append(df_long)

df_all = pd.concat(dfs, ignore_index=True)
df_all["SZA_bin_end"] = df_all["SZA_bin"].str.split("-").str[1].astype(float)



# FacetGrid ordered by altitude
g = sns.FacetGrid(
    df_all,
    col="Site",
    col_wrap=3,
    col_order=site_order,
    height=4,    # slightly bigger subplots
    sharey=True
)

#g.map_dataframe(sns.lineplot, x="SZA_bin", y="rRMSE", hue="Model", marker="o")
g.map_dataframe(sns.lineplot, x="SZA_bin_end", y="rRMSE", hue="Model", marker="o")
g.set_axis_labels("SZA bin end (°)", "rRMSE")
g.set(ylim=(0, 90))
g.add_legend(loc="upper center", ncols=4 )
#g.set_axis_labels("SZA bin (°)", "rRMSE")
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.05, right=0.95, hspace=0.3, wspace=0.05)
#plt.subplots_adjust(top=0.9)
#g.fig.suptitle("Comparison of rRMSE vs SZA (ordered by altitude)", fontsize=16)

plt.show()



