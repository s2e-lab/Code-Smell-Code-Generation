import json

with open('HumanEval.jsonl') as file:
    count = 0
    for line in file:
        data = json.loads(line)
        with open('./HumanEval/'+str(count)+'.py', 'w') as f:
            f.write(data['prompt'])
        count += 1
