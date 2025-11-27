import pandas as pd
import Metrics as ms
site = 'YU'
dTest = pd.read_csv(f'test_{site.lower()}_15.csv')

print(ms.rrmsd(dTest.ghi, dTest.lsasaf))