import pandas as pd

dfEro = pd.read_csv('ero15.csv')
dfLQ = pd.read_csv('lq15.csv')

dfEro.columns
dfLQ.columns


import matplotlib.pyplot as plt

# Definimos las columnas que queremos graficar
cols = ["ghi", "cams", "lsasaf"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ----------- Primer subplot: dfEro -----------
for col in cols:
    axes[0].hist(dfEro[col], bins=1000, alpha=0.5, label=col)

axes[0].set_title("dfEro - Histogramas")
axes[0].set_xlabel("Valor")
axes[0].set_ylabel("Frecuencia")
axes[0].legend()

# ----------- Segundo subplot: dfLQ -----------
for col in cols:
    axes[1].hist(dfLQ[col], bins=1000, alpha=0.5, label=col)

axes[1].set_title("dfLQ - Histogramas")
axes[1].set_xlabel("Valor")
axes[1].legend()

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

# Columnas a comparar
cols = ["ghi", "cams", "lsasaf"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ----------- Primer subplot: dfEro -----------
for col in cols:
    sns.kdeplot(dfEro[col], ax=axes[0], label=col, fill=False)

axes[0].set_title("dfEro - Distribuciones KDE")
axes[0].set_xlabel("Valor")
axes[0].set_ylabel("Densidad")
axes[0].legend()

# ----------- Segundo subplot: dfLQ -----------
for col in cols:
    sns.kdeplot(dfLQ[col], ax=axes[1], label=col, fill=False)

axes[1].set_title("dfLQ - Distribuciones KDE")
axes[1].set_xlabel("Valor")
axes[1].legend()

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

# Columnas que queremos comparar
cols = ["ghi", "cams", "lsasaf"]

# Crear figura con 2 subplots (uno para cada DataFrame)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ----------- Primer subplot: dfEro -----------
for col in cols:
    sns.kdeplot(
        dfEro[col],
        ax=axes[0],
        label=col,
        fill=False,
        cut=0 # <-- evita extenderse fuera del rango de los datos
    )

axes[0].set_title("dfEro - Distribuciones KDE")
axes[0].set_xlabel("Valor")
axes[0].set_ylabel("Densidad")
axes[0].set_xlim(left=0)  # <-- asegura que no muestre negativos
axes[0].legend()

# ----------- Segundo subplot: dfLQ -----------
for col in cols:
    sns.kdeplot(
        dfLQ[col],
        ax=axes[1],
        label=col,
        fill=False,
        cut=0
    )

axes[1].set_title("dfLQ - Distribuciones KDE")
axes[1].set_xlabel("Valor")
axes[1].set_xlim(left=0)
axes[1].legend()

plt.tight_layout()
plt.show()
