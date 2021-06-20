import numpy as np
import pandas as pd
import os

target_dict = {
	'V0' : [0, 1, 2, 3, 4],
	'V2' : [0, 1, 2, 4],
	'V3' : [0, 1, 2, 4],
	'V4' : [0, 1, 2, 4],
}

base_dir = r'S:\DataSets\cassavaVersion\V5\V5-4'

results = None
for k, v in target_dict.items():
	csv_path = os.path.join(base_dir, k, 'train.csv')
	print(csv_path)
	df = pd.read_csv(csv_path)
	for label in v:
		choises = df[df.label == label].to_numpy()
		if len(choises) <= 0:
			continue
		print(choises.shape)
		if results is None:
			results = choises
		else:
			results = np.vstack([results, choises])

print(results.shape)
sub = pd.DataFrame(data=results, index=None, columns=['image_id', 'label'])
sub.to_csv(os.path.join(base_dir, 'train.csv'), index=False)
print('||' * 30)
df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
print(df.label.value_counts())