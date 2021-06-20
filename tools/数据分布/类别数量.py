import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv(r'S:\DataSets\cassavaVersion\V5\V5-3\2019extra.csv')
print(df.label.value_counts())