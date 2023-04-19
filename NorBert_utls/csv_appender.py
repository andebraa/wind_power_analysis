import pandas as pd 

file1 = pd.read_csv('anotized_data_100.csv') 
file2 = pd.read_csv('anotized_data_100_2.csv') 

out = file1.append(file2) 

out.to_csv('anotized_combined_200.csv', index = False)
