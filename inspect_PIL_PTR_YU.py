import pandas as pd

dfBRB = pd.read_csv('pil15.csv')
dfSA = pd.read_csv('ptr5.csv')
dfSCA = pd.read_csv('yu15.csv')



import matplotlib.pyplot as plt
import seaborn as sns

# Columnas que queremos comparar
cols = ["ghi", "cams", "lsasaf"]

# Crear figura con 2 subplots (uno para cada DataFrame)
fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

# ----------- Primer subplot: dfEro -----------
for col in cols:
    sns.kdeplot(
        dfBRB[col],
        ax=axes[0],
        label=col,
        fill=False,
        cut=0 # <-- evita extenderse fuera del rango de los datos
    )

axes[0].set_title("PIL")
axes[0].set_xlabel("Valor")
axes[0].set_ylabel("Densidad")
axes[0].set_xlim(left=0)  # <-- asegura que no muestre negativos
axes[0].legend()

# ----------- Segundo subplot: dfLQ -----------
for col in cols:
    sns.kdeplot(
        dfSA[col],
        ax=axes[1],
        label=col,
        fill=False,
        cut=0
    )

axes[1].set_title("PTR")
axes[1].set_xlabel("Valor")
axes[1].set_xlim(left=0)
axes[1].legend()


for col in cols:
    sns.kdeplot(
        dfSCA[col],
        ax=axes[2],
        label=col,
        fill=False,
        cut=0
    )

axes[2].set_title("YU")
axes[2].set_xlabel("Valor")
axes[2].set_xlim(left=0)
axes[2].legend()


plt.tight_layout()
plt.show()
