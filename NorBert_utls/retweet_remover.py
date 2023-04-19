import pandas as pd 

data = pd.read_csv('anotize_data/ready_for_use/anotized_data_comb_2_3_4_5.csv', usecols = ['labels', 'text']) 

mask = data.text.str.contains(r"^RT")

data_no_rt = pd.DataFrame()
data_no_rt = data[~mask]

data_no_rt.to_csv('anotize_data/ready_for_use/all_anotized_data_no_rt.csv', index = False)


