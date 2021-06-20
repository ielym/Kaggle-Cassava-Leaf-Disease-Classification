import numpy as np
import pandas as pd
import os
from glob import glob

category_id = {
				'cbb':0,
				'cbsd': 1,
				'cgm':2,
				'cmd':3,
				'healthy':4,
			}

id_category = {
	0: "cbb",
	1: "cbsd",
	2: "cgm",
	3: "cmd",
	4: "healthy",
}

base_dir = r'S:\DataSets\cassava-disease-2019-train\train'

child_dirs = os.listdir(base_dir)

img_name_id = []
for child_dir in child_dirs:
	if child_dir not in category_id.keys():
		continue
	print(child_dir)
	category = category_id[child_dir]
	child_dir_path = os.path.join(base_dir, child_dir)
	files_names = os.listdir(child_dir_path)
	print(len(files_names))
	for file_name in files_names:
		# img_name_id.append(['{}/{}'.format(child_dir, file_name), category])
		img_name_id.append(['{}'.format(file_name), category])

df = pd.DataFrame(data=img_name_id, index=None, columns=['image_id', 'label'])
df.to_csv(os.path.join(base_dir, 'train.csv'), index=False)
