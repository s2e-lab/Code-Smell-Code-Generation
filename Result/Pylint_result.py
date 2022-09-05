import json
from collections import defaultdict
import os
import csv

filters = ['C0303', 'C0304', 'C0305', 'C0103', 'C0112', 'C0114', 'C0115', 'C0116', 'W0611', 'C0415', 'W0404',
           'C0413', 'C0411', 'C0412', 'W0401', 'W0614', 'C0410', 'C0413', 'C0414', 'R0402', 'C2403', 'E0401', 'E0001']

smellList = []
smellDict = defaultdict()
smellDictName = defaultdict()

dirPath = './pylint_data/'
for file in os.listdir(dirPath):
    # check only json files
    if file.endswith('.json'):
        # print(file)
        with open(dirPath+file) as f:
            data = json.load(f)
            # print(data)
            for items in data:
                for item in items:
                    smellList.append(item)
                    found = False
                    for filter in filters:
                        if item["message-id"] == filter:
                            found = True
                        if item["message-id"].startswith('F'):
                            found = True
                    if found:
                        continue
                    try:
                        # smellDict[item["message-id"]] += 1
                        smellDictName[item["message-id"] +
                                      "-"+item["symbol"]] += 1
                    except KeyError:
                        # smellDict[item["message-id"]] = 1
                        smellDictName[item["message-id"] +
                                      "-"+item["symbol"]] = 1

data = []
for key in smellDictName:
    print(key)
    row = []
    row.append(key)
    row.append(smellDictName[key])
    data.append(row)
header = ['Message', 'Count']

with open('dataset.csv', 'w',  newline='', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerows(data)
