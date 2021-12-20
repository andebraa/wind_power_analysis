import pandas as pd 
import numpy as np

data = pd.read_csv('final_dataset_boollabel.csv')

print(np.sum(data['labels'] == 1))
