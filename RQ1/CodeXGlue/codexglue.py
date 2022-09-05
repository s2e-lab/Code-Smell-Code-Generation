import json
import os
basedir = 'codexglue/'  # change this to the path of your target directory
with open('train.jsonl') as file:
    lines = file.readlines()
    lines = [json.loads(line) for line in lines]
    count = 0
    directory = ''
    for line in lines:
        if count % 1000 == 0:
            directory = str(count)
            path = os.path.join(basedir, directory)
            os.mkdir(path)
        try:
            with open(basedir+directory+'/'+str(count)+'.py', 'w') as file:
                file.write(line['original_string'])
        except:
            print('error occured')
        count += 1
