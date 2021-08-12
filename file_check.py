

file = open('temp_citydata.csv', 'r')
non_empts = 0

for line in file:
    elem = line.split(',')
    
    if elem[1] != '\n':
        non_empts += 1
        print(elem)
print(non_empts)
