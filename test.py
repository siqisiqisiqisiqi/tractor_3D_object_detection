import json
import glob
import matplotlib.pyplot as plt
import re


# json_file_list = glob.glob("./results/0807-2025*/*.json")
# split_json_file_list = [s.split("/")[3] for s in json_file_list]
# numbers = [int(re.findall(r'\d+', s)[0]) for s in split_json_file_list]
# sorted_indices = sorted(range(len(numbers)), key=lambda k: numbers[k])

# iou_value_list = []
# for i in sorted_indices:
#     file = json_file_list[i]
#     with open(file, 'r') as json_file:
#         loaded_dict = json.load(json_file)
#         iou_value = loaded_dict['test_iou3d']
#         iou_value_list.append(iou_value)

# plt.plot(iou_value_list)
# plt.grid()
# plt.show()
