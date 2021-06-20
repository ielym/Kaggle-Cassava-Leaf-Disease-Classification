import numpy as np
import pandas as pd
import os

#27053

df1 = pd.read_csv(r'S:\DataSets\cassava-disease-2019-train\train\train.csv')
df2 = pd.read_csv(r'S:\DataSets\cassava-leaf-disease-classification\train.csv')

labels1 = df1.to_numpy()
labels2 = df2.to_numpy()

merge_labels = np.vstack([labels1, labels2])
print(labels1.shape, labels2.shape, merge_labels.shape)

sub = pd.DataFrame(data=merge_labels, index=None, columns=['image_id', 'label'])
sub.to_csv(r'./train.csv', index=False)

df = pd.read_csv(r'./train.csv')
print(df.shape)