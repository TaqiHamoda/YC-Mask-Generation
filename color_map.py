import random 
import numpy as np
import csv
import matplotlib.colors as mcolors
# f = open('mapping.txt','r')
# csv_file = open('color_mapping.csv',mode='w',encoding='utf-8',newline="")
# writer = csv.writer(csv_file)
# for line in f.readlines():
#     content = line.strip()
#     colors = [random.randint(0, 255) for _ in range(3)]
#     colors = np.random.rand(1,3)
#     hex_color = mcolors.to_hex(colors)
#     # print(hex)
#     writer.writerow([str(content),str(hex_color)])

# colors = [random.randint(0, 255) for _ in range(3)]
# colors = np.random.rand(1,3)
# hex_color = mcolors.to_hex(colors)
# print(hex_color)
f = open('mapping.txt','r')
label_list = []
for line in f.readlines():
    label_list.append(line.strip())
f.close()

f = open('seedpoints_on_images.csv','r')
reader = csv.reader(f)
map_list=[]
print(label_list)
for row in reader:    
    img_name,x,y,cl = row
    
    if cl not in label_list:
        print(cl)
        label_list.append(cl)
        wf = open('mapping.txt','a')
        new_line = f'{cl}\n'
        wf.write(line)
        wf.close()
