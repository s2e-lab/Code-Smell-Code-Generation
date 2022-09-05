import json
from collections import defaultdict
import csv


smellList = []
smellDict = defaultdict()
smellDictCWE = defaultdict()
smellPath = defaultdict()
error = 0
analyzed = 0
with open('dataset.json') as f:
    data = json.load(f)
    for item in data['results']:
        smellList.append(item)
        try:
            smellDict[item["test_id"]+'_'+item["test_name"]] += 1
            # smellDictCWE[item["issue_cwe"]["id"]] += 1
            #smellPath[item['filename']] += 1
        except KeyError:
            #print("KeyError:", item["test_id"],item["test_name"])
            smellDict[item["test_id"]+'_'+item["test_name"]] = 1
            # smellDictCWE[item["issue_cwe"]["id"]] = 1
            #smellPath[item['filename']] = 1
data = []
for key in smellDict:
    print(key)
    row = []
    row.append(key)
    row.append(smellDict[key])
    data.append(row)
header = ['Message', 'Count']

with open('dataset_bandit.csv', 'w',  newline='', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerows(data)
