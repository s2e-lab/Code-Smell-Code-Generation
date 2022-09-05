import glob
from hashlib import new
import os

basepath = './ParsedTrainFiles/'  # change this to the path of your target directory
text_files = glob.glob("./APPS/train/**/solutions.json", recursive=True)
text_files.sort()

for i in range(len(text_files)):

    # if not "0001" in text_files[i]:
    #     continue
    # print(text_files[i])
    file = open(text_files[i], "r")
    comma_split = file.read()
    comma_split = comma_split[1:-1]
    comma_split = comma_split[1:-1]
    # print(comma_split)
    dir_name = os.path.split(os.path.dirname(text_files[i]))[1]
    print(dir_name)
    newpath = basepath+dir_name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # file_write = open("./train_py_files/" + dir_name + ".py", "w")
    comma_split = comma_split.split("\", \"")
    # print(len(comma_split))
    for j in range(len(comma_split)):
        file_write = open(newpath+'/' + dir_name+'_'+str(j) + ".py", "w")
        comma_split[j] = comma_split[j].replace('\\n', '\n')
        comma_split[j] = comma_split[j].replace('\\t', '\t')

        comma_split[j] = comma_split[j].replace('\\', '')
        # print(comma_split[j])
        # print('\n')
        file_write.write(comma_split[j])
        file_write.write('\n')
