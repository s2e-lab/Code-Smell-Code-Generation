import os
import json

foldername = ['gpt-code-clippy-125M-1024-f', 'gpt-neo-125M-code-clippy-code-search-py', 'gpt-neo-125M', 'gpt-neo-125M-code-clippy-dedup-2048', 'gpt-neo-125M-apps',
              'gpt-neo-125M-code-clippy-dedup-filtered-no-resize-2048bs', 'gpt-neo-125M-code-clippy', 'gpt-neo-125M-code-search-all', 'gpt-neo-125M-code-clippy-code-search-all', 'gpt-neo-125M-code-search-py']
for folder in foldername:
    print(folder)
    jsonfilelocation = "./gpt-code-clippy-evaluation-model_results/" + \
        folder+"/human_eval.jsonl_results.jsonl"
    outputfolderlocation = "./gpt-code-clippy-evaluation-model_results/"+folder+"/files/"
    print(jsonfilelocation)
    print(outputfolderlocation)
    direc = os.getcwd() + "/" + outputfolderlocation
    if not os.path.exists(direc):
        os.makedirs(direc)
        print("directory \'" + direc + "\' created.")

    jsonfile = open(jsonfilelocation, "r", encoding="utf8", errors='ignore')

    # load the input files
    jsonlines = jsonfile.readlines()
    count = 1
    for line in jsonlines:
        jsonline = json.loads(line)
        original_string = jsonline["completion"]
        code = original_string
        tempfile = open(outputfolderlocation + "func" +
                        str(count + 1) + ".py", "w")
        # tempfile = open(filename + str(lineRead + 1) + "c" + str(chunk) + ".py", "w")
        code = ascii(code)
        code = code[1:-1]
        code = code.replace('\\n', '\n')
        code = code.replace('\\t', '\t')

        code = code.replace('\\', '')
        tempfile.write((code))
        tempfile.close()
        count += 1

    # input files close
    jsonfile.close()
