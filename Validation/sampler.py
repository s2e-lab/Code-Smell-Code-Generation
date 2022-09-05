import random
from collections import defaultdict
import json

# key: message-id, value: list of instance of this message found in the dataset
instances = defaultdict(list)

sampleList = []
total = 0

# it is a variable to control the highest number of instances of a message
HIGHEST_INSTANCE = 2

for key in instances:
    if len(instances[key]) >= HIGHEST_INSTANCE:
        instances[key] = random.instances(instances[key], HIGHEST_INSTANCE)
    else:
        instances[key] = instances[key]
    total += len(instances[key])
    sampleList.extend(instances[key])

print(total)

# it is a variable to control the sample size
targetSampleSize = 257
if total >= targetSampleSize:
    sampleList = random.instances(sampleList, targetSampleSize)
jsonData = json.dumps(sampleList)
jsonFile = open("analyzer_dataset.json", "w")
jsonFile.write(jsonData)
jsonFile.close()
