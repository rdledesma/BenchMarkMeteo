import pandas as pd
import matplotlib.pyplot as plt
import datetime
d  = pd.read_csv('ero15.csv')
d['datetime'] = pd.to_datetime(d.datetime)


plt.figure()
plt.plot(d.datetime, d.ghi, '.-r', label="Medida")
plt.plot(d.datetime, d.lsasaf, '.-b', label="LSA-SAF")
plt.plot(d.datetime, d.cams, '.-g', label="CAMS")
plt.legend()
plt.show()

d20161012 = d[d.datetime.dt.date == datetime.date(2016,10,12)]

d20170503 =  d[d.datetime.dt.date == datetime.date(2017,5,3)]

plt.figure()
plt.plot(d20161012.datetime, d20161012.ghi, '.-r', label="Medida")
plt.plot(d20161012.datetime, d20161012.lsasaf, '.-b', label="LSA-SAF")
plt.plot(d20161012.datetime, d20161012.cams, '.-g', label="CAMS")
plt.legend()
plt.show()

plt.figure()
plt.plot(d20170503.datetime, d20170503.ghi, '.-r', label="Medida")
plt.plot(d20170503.datetime, d20170503.lsasaf, '.-b', label="LSA-SAF")
plt.plot(d20170503.datetime, d20170503.cams, '.-g', label="CAMS")
plt.legend()
plt.show()




import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

# Primer día
plt.plot(d20161012.datetime, d20161012.ghi, '.-r', label="Medida 2016-10-12")
plt.plot(d20161012.datetime, d20161012.lsasaf, '.-b', label="LSA-SAF 2016-10-12")
plt.plot(d20161012.datetime, d20161012.cams, '.-g', label="CAMS 2016-10-12")

# Segundo día
plt.plot(d20170503.datetime, d20170503.ghi, 'o--r', label="Medida 2017-05-03")
plt.plot(d20170503.datetime, d20170503.lsasaf, 'o--b', label="LSA-SAF 2017-05-03")
plt.plot(d20170503.datetime, d20170503.cams, 'o--g', label="CAMS 2017-05-03")

plt.legend()
plt.xlabel("Tiempo")
plt.ylabel("GHI")
plt.title("Comparación de GHI, LSA-SAF y CAMS en dos días")
plt.show()
3



import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12,8))

# Primer día
axes[0].plot(d20161012.datetime, d20161012.ghi, '.-r', label="Medida")
axes[0].plot(d20161012.datetime, d20161012.lsasaf, '.-b', label="LSA-SAF")
axes[0].plot(d20161012.datetime, d20161012.cams, '.-g', label="CAMS")
axes[0].set_title("2016-10-12")
axes[0].legend()

# Segundo día
axes[1].plot(d20170503.datetime, d20170503.ghi, '.-r', label="Medida")
axes[1].plot(d20170503.datetime, d20170503.lsasaf, '.-b', label="LSA-SAF")
axes[1].plot(d20170503.datetime, d20170503.cams, '.-g', label="CAMS")
axes[1].set_title("2017-05-03")
axes[1].legend()

plt.xlabel("Tiempo")
plt.ylabel("GHI")
plt.tight_layout()
plt.show()







import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)

# Primer día
axes[0].plot(d20161012.datetime, d20161012.ghi, '.-r', label="Medida")
axes[0].plot(d20161012.datetime, d20161012.lsasaf, '.-b', label="LSA-SAF")
axes[0].plot(d20161012.datetime, d20161012.cams, '.-g', label="CAMS")
axes[0].set_title("2016-10-12")
axes[0].set_xlabel("Tiempo")
axes[0].set_ylabel("GHI")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=45)  # rota las fechas

# Segundo día
axes[1].plot(d20170503.datetime, d20170503.ghi, '.-r', label="Medida")
axes[1].plot(d20170503.datetime, d20170503.lsasaf, '.-b', label="LSA-SAF")
axes[1].plot(d20170503.datetime, d20170503.cams, '.-g', label="CAMS")
axes[1].set_title("2017-05-03")
axes[1].set_xlabel("Tiempo")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
