# An Empirical Study of Code Smells in Transformer-based Code Generation Techniques

This repository contains the scripts used for the accepted paper in the research track of the 22nd IEEE International Working Conference on Source Code Analysis and Manipulation (SCAM 2022), titled **An Empirical Study of Code Smells in Transformer-based Code Generation Techniques**. The paper provides an empirical study on containing (security) code smell in the code generation dataset and the output of the code generation model. It also demonstrates the code smell in the output of GitHub Copilot. The pre-print can be found here: [Preprint](scam_2022.pdf)

## Installation
We used Pylint and Bandit for our analysis.
To install pylint, run the following command:
```
pip install pylint
```
To install bandit, run the following command:
```
pip install bandit
```
In both cases, you may create a virtual environment and install the packages in it.


## Research Questions

![Methodology](https://github.com/s2e-lab/Code-Smell-Code-Generation/blob/main/Methodology.png?raw=true)


### RQ1: Are code smells present in the code generation training datasets?

We used three datasets in our project:
1. [APPS](https://github.com/hendrycks/apps):
   1. Download and unzipthe dataset in ```RQ1/APPS/``` folder from https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
   2. Create a folder, ```RQ1/APPS/ParsedTrainFiles/```
   3. Use this [script](/RQ1/APPS/apps.py) to parse all the solution files for the train set of the dataset. [Data](RQ1/APPS/ParsedTrainFiles/) The output can be found here: [Pylint](/RQ1/APPS/Pylint/), [Bandit](/RQ1/APPS/bandit.json).
2. [CodeXGlue](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text): 
   1. Use this [script](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text#download-data-and-preprocess) mentioned in the CodeXGlue repository to download the dataset 
   2. Used this [script](/RQ1/CodeXGlue/codexglue.py) to parse all the code snippets for the train set of the dataset. We created a bucket of 1000 samples to ease the analysis. The output can be found here: [Data](RQ1/CodeXGlue/dataset), [Pylint](/RQ1/CodeXGlue/pylint_data), [Bandit](/RQ1/CodeXGlue/bandit.json).
3. [CodeClippy](https://the-eye.eu/public/AI/training_data/code_clippy_data/code_clippy_dedup_data/train/): We first used this [script](/RQ1/Code_Clippy/code_clippy.py) to download all the code snippets from the duplication free training set and get only parsed python samples after creating ```zsts``` and ```pyjsons``` folders. Then, we used this [script](/RQ1/Code_Clippy/code_clippy_pylint.py) to analyze the code snippets and get the code smells using Pylint. The output can be found here: [output](/RQ1/Code_Clippy/).

The following libraries need to install first to run script for CodeClippy:
 ```
 pip install python-magic
 pip install zstandard
 pip install colorama
 pip install BeautifulSoup4
 ```
If you face problem with libmagic, you can check this: https://github.com/Yelp/elastalert/issues/1927#issuecomment-689927231


We used this [script](/RQ1/PylintRunner.py) for the first two datasets to run Pylint on the code snippets and get the code smells. The usage of this script is as follows:
```
python PylintRunner.py ./APPS/ParsedTrainFiles 
python PylintRunner.py ./CodeXGlue/dataset
```

To run bandit, we use the following command:
```
bandit -r ./APPS/ParsedTrainFiles  -f json -o ./APPS/bandit.json
bandit -r ./CodeXGlue/dataset  -f json -o ./CodeXGlue/bandit.json
```


### RQ2: Does the output of an open-source transformer-based code generation technique contain code smells?
We downloaded the output of the GPT-Neo Model's various configurations from here: https://github.com/CodedotAl/gpt-code-clippy/tree/camera-ready/evaluation/model_results. We used this [script](/RQ2/parser.py) to parse the output JSONL into an individual python sample. Then, we run Pylint and Bandit on them using the previous scripts mentioned in RQ1. The output can be found here: [output](/RQ2/CodeClippyOutput/).

Move PylintRunner.py to ```RQ2``` folder and run the following command:
```
python PylintRunner.py ./gpt-code-clippy-evaluation-model_results/gpt-code-clippy-125M-1024-f/files
bandit -r ./gpt-code-clippy-evaluation-model_results/gpt-code-clippy-125M-1024-f/files -f json -o bandit.json
```
Move the file to appropriate folder.


### RQ3: Is there any code smell in the output of closed source code generation tools based on a large language model?
We manually generated output from GitHub Copilot using the HumanEval dataset after parsing the JSONL file into an individual python sample using this [script](/RQ3/CopilotOutput/HumanEval/parser.py) and then run Pylint and Bandit on it. The generated code and output can be found here: [output](/RQ3/CopilotOutput/).

## Result Generation

To get pylint result, we use this [pylint_script](/Result/Pylint_result.py) and [bandit_script](/Result/Bandit_result.py) to get the pylint and bandit result for the three datasets, 10 configurations's output and Copilot's output in a CSV format. The final output can be found here: [Result](/Result/)

Then, you may run [TableGenerato](/Result/TableGeneration.py) to find out the result in a table format. 

## Validation
Finally, we validated the output of our two analyzers: Pylint and Bandit. We used this [script](/Validation/sampler.py) to collect the sample. The final sample list can be found here: [output](/Validation/)


## Abstract

Prior works have developed transformer-based language learning models to automatically generate source code for a task without compilation errors. The datasets used to train these techniques include samples from open source projects which may not be free of security flaws, code smells, and violations of standard coding practices. Therefore, we investigate to what extent code smells are present in the datasets of coding generation techniques and verify whether they leak into the output of these techniques. To conduct this study, we used Pylint and Bandit to detect code smells and security smells in three widely used training sets (CodeXGlue, APPS, and Code Clippy). We observed that Pylint caught 264 code smell types, whereas Bandit located 44 security smell types in these three datasets used for training code generation techniques. By analyzing the output from ten different configurations of the open-source fine- tuned transformer-based GPT-Neo 125M parameters model, we observed that this model leaked the smells and non-standard practices to the generated source code. When analyzing GitHub Copilot’s suggestions, a closed source code generation tool, we observed that it contained 18 types of code smells, including substandard coding patterns and 2 security smell types.

## Citation
Please do cite the paper, if you use the code or the paper in your work.

```
@inproceedings{siddiq2022empirical,
  author={Siddiq, Mohammed Latif and Majumder, Shafayat Hossain and Mim, Maisha Rahman and Jajodia, Sourov and Santos, Joanna C. S. },
  booktitle={Proceedings of the 22nd International Working Conference on Source Code Analysis and Manipulation (SCAM)}, 
  title={An Empirical Study of Code Smells in Transformer-based Code Generation Techniques}, 
  year={2022},
  month={Oct},
  doi={}
}

```