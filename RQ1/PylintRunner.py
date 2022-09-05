from pylint import epylint as elint
import os
import sys

import json

if len(sys.argv) < 2:
    print("not enough arguments")
    exit()
basedir = sys.argv[1]
if basedir[-1] != '/':
    basedir += '/'
outputDir = basedir+'../pylint_data'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
    print("directory \'" + outputDir + "\' created.")
totalCount = 0
result = []
for (root, dirs, file) in os.walk(basedir):
    for f in file:
        # print(f)
        try:
            pylint_stdout, pylint_stderr = elint.py_run(
                root+'/'+f + ' --output-format=json', return_std=True)
            # print(pylint_stdout.getvalue())

            result.append(json.loads(pylint_stdout.getvalue()))
        except:
            print('error in file: ' + f)
        totalCount += 1
        if totalCount % 100 == 0:
            print(totalCount)
            with open(outputDir+'/'+str(totalCount)+'.json', 'w') as f:
                f.write(json.dumps(result))
            f.close()
            result = []


if len(result) > 0:
    with open(outputDir+'/'+str(totalCount)+'.json', 'w') as f:
        f.write(json.dumps(result))
    f.close()
    result = []
