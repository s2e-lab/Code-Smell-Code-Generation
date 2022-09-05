def get_chinese_text():
    """Download the chinese_text dataset and unzip it"""
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if (not os.path.exists('data/pos.txt')) or \
       (not os.path.exists('data/neg')):
        os.system("wget -q https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/chinese_text.zip "
                  "-P data/")
        os.chdir("./data")
        os.system("unzip -u chinese_text.zip")
        os.chdir("..")