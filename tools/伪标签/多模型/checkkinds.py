import numpy as np
import pandas as pd

df = pd.read_csv(r'S:\DataSets\cassavaVersion\V1\train.csv').to_numpy()

count = 0
for line in df:
	if line[1] == 3:
		count += 1
print(count)
