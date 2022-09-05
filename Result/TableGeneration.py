# %%
import csv


# %% [markdown]
# Table 1 Generation

# %%

filenames = ["apps","clippy","codexglue"]
for filename in filenames:
    print(filename)
    e,c,r,w = 0,0,0,0
    with open("./RQ1_Result/Pylint/"+filename+".csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            if(row[0].startswith('E')):
                e+=int(row[1])
            elif(row[0].startswith('C')):
                c+=int(row[1])
            elif(row[0].startswith('R')):
                r+=int(row[1])
            elif(row[0].startswith('W')):
                w+=int(row[1])
    print("E: ",e)
    print("C: ",c)
    print("R: ",r)
    print("W: ",w)
    print("Total: ",e+c+r+w)
    if filename == "apps":
        print("Avg: ",(e+c+r+w)/117232)
    elif filename == "clippy":
        print("Avg: ",(e+c+r+w)/139655)
    elif filename == "codexglue":
        print("Avg: ",(e+c+r+w)/251820)

# %% [markdown]
# Table 2 Generation

# %%
from collections import defaultdict
import json

# %%
pylint = defaultdict(list)
filenames = ["apps","clippy","codexglue"]
for filename in filenames:
    print(filename)
    with open("./RQ1_Result/Pylint/"+filename+".csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            if filename == "apps":
                pylint[row[0]].append(int(row[1])/117232)
            elif filename == "clippy":
                pylint[row[0]].append(int(row[1])/139655)
            elif filename == "codexglue":
                pylint[row[0]].append(int(row[1])/251820)            
numDict = defaultdict(int)
for key in pylint:
    numDict[key] = sum(pylint[key])
    pylint[key].append(sum(pylint[key]))
jsonData = json.dumps(pylint)
jsonFile = open("pylint_result_modified.json", "w")
jsonFile.write(jsonData)
jsonFile.close()
numDict = dict(sorted(numDict.items(), key=lambda item: item[1]))
jsonData = json.dumps(numDict)

jsonFile = open("pylint_result_total.json", "w")
jsonFile.write(jsonData)
jsonFile.close()

# %% [markdown]
# Table 3 and 4 Generation

# %%

filenames = ["apps","codeclippy","codexglue"]
for filename in filenames:
    print(filename)
    total = 0
    total_type = 0
    bandit = defaultdict(int)

    with open("./RQ1_Result/Bandit/"+filename+"_bandit.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            total+=int(row[1])
            total_type+=1
            bandit[row[0]] = int(row[1])
    
    print("Total: ",total)
    print("Total Type: ",total_type)
    if filename == "apps":
        print("Avg: ",(total)/117232)
    elif filename == "codeclippy":
        print("Avg: ",(total)/139655)
    elif filename == "codexglue":
        print("Avg: ",(total)/251820)
    numDict = dict(sorted(bandit.items(), key=lambda item: item[1]))
    print(numDict)

# %% [markdown]
# Table 5 Generation

# %%

filenames = ['gpt-code-clippy-125M-1024-f', 'gpt-neo-125M-code-clippy-code-search-py', 'gpt-neo-125M', 'gpt-neo-125M-code-clippy-dedup-2048', 'gpt-neo-125M-apps',
              'gpt-neo-125M-code-clippy-dedup-filtered-no-resize-2048bs', 'gpt-neo-125M-code-clippy', 'gpt-neo-125M-code-search-all', 'gpt-neo-125M-code-clippy-code-search-all', 'gpt-neo-125M-code-search-py']
for filename in filenames:
    print(filename)
    e,c,r,w = 0,0,0,0
    with open("./RQ2_Final/Pylint/"+filename+".csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            if(row[0].startswith('E')):
                e+=int(row[1])
            elif(row[0].startswith('C')):
                c+=int(row[1])
            elif(row[0].startswith('R')):
                r+=int(row[1])
            elif(row[0].startswith('W')):
                w+=int(row[1])
    print("E: ",e)
    print("C: ",c)
    print("R: ",r)
    print("W: ",w)
    print("Total: ",e+c+r+w)
    print("Avg: ",(e+c+r+w)/1640)

# %% [markdown]
# Table 6 Generation

# %%
pylint = defaultdict(list)
filenames = ['gpt-code-clippy-125M-1024-f', 'gpt-neo-125M-code-clippy-code-search-py', 'gpt-neo-125M', 'gpt-neo-125M-code-clippy-dedup-2048', 'gpt-neo-125M-apps',
              'gpt-neo-125M-code-clippy-dedup-filtered-no-resize-2048bs', 'gpt-neo-125M-code-clippy', 'gpt-neo-125M-code-search-all', 'gpt-neo-125M-code-clippy-code-search-all', 'gpt-neo-125M-code-search-py']
for filename in filenames:
    print(filename)
    with open("./RQ2_Final/Pylint/"+filename+".csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            pylint[row[0]].append(int(row[1]))          
numDict = defaultdict(int)
for key in pylint:
    numDict[key] = len(pylint[key])
    pylint[key].append(sum(pylint[key]))
jsonData = json.dumps(pylint)
jsonFile = open("pylint_result_modified_rq2.json", "w")
jsonFile.write(jsonData)
jsonFile.close()
numDict = dict(sorted(numDict.items(), key=lambda item: item[1]))
jsonData = json.dumps(numDict)

jsonFile = open("pylint_result_total_rq2.json", "w")
jsonFile.write(jsonData)
jsonFile.close()

# %% [markdown]
# Table 7 Generation

# %%

filenames = ['gpt-code-clippy-125M-1024-f', 'gpt-neo-125M-code-clippy-code-search-py', 'gpt-neo-125M', 'gpt-neo-125M-code-clippy-dedup-2048', 'gpt-neo-125M-apps',
              'gpt-neo-125M-code-clippy-dedup-filtered-no-resize-2048bs', 'gpt-neo-125M-code-clippy', 'gpt-neo-125M-code-search-all', 'gpt-neo-125M-code-clippy-code-search-all', 'gpt-neo-125M-code-search-py']
for filename in filenames:
    print(filename)
    total = 0
    total_type = 0
    bandit = defaultdict(int)

    with open("./RQ2_Final/Bandit/"+filename+".csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            total+=int(row[1])
            total_type+=1
            bandit[row[0]] = int(row[1])
    
    print("Total: ",total)
    print("Total Type: ",total_type)

    numDict = dict(sorted(bandit.items(), key=lambda item: item[1]))
    print(numDict)
    print()

# %% [markdown]
# Table 8 Generation

# %%

for i in range(4):
    print(i)
    e,c,r,w = 0,0,0,0
    with open("./RQ3_Final/Pylint/HumanEval_"+str(i)+".csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            if(row[0].startswith('E')):
                e+=int(row[1])
            elif(row[0].startswith('C')):
                c+=int(row[1])
            elif(row[0].startswith('R')):
                r+=int(row[1])
            elif(row[0].startswith('W')):
                w+=int(row[1])
    print("E: ",e)
    print("C: ",c)
    print("R: ",r)
    print("W: ",w)
    print("Total: ",e+c+r+w)
    print("Avg: ",(e+c+r+w)/164)

# %% [markdown]
# Table 9 Generation

# %%
pylint = defaultdict(list)
for i in range(4):
    e,c,r,w = 0,0,0,0
    with open("./RQ3_Final/Pylint/HumanEval_"+str(i)+".csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if(row[0]=='Message'):
                continue
            pylint[row[0]].append(int(row[1]))          

print(pylint.keys())


