'''Firstly You have to install magic with &pip install python-magic& and zstandard with &pip install zstandard&'''
import magic
import os
import json
import uuid
import zstandard
import subprocess
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import colorama

colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW

internal_urls = set()
external_urls = set()


def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_website_links(url):
    """
    Returns all URLs that is found on `url` in which it belongs to the same website
    """
    # all URLs of `url`
    urls = set()
    # domain name of the URL without the protocol
    domain_name = urlparse(url).netloc
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            # href empty tag
            continue
    # join the URL if it's relative (not absolute link)
        href = urljoin(url, href)
        parsed_href = urlparse(href)
    # remove URL GET parameters, URL fragments, etc.
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if not is_valid(href):
            # not a valid URL
            continue
        if href in internal_urls:
            # already in the set
            continue
        if domain_name not in href:
            # external link
            if href not in external_urls:
                print(f"{GRAY}[!] External link: {href}{RESET}")
                external_urls.add(href)
                continue
        # print(f"{GREEN}[*] Internal link: {href}{RESET}")
        urls.add(href)
        internal_urls.add(href)
    return urls


def loadJsonL(fname):
    import json

    data = []
    with open(fname) as fp:
        for line in fp.readlines():
            data.append(json.loads(line))
    return data


def processZSTLink(url, dir, outputdir):
    zstfile = url.split('/')[-1]
    print("downloading: ", url)
    dir += zstfile
    out = subprocess.run(
        f"wget -O {dir} -q {url} ", shell=True, stdout=subprocess.DEVNULL)
    # dir = dir[:-4]
    outputdir += zstfile[:-4]
    with open(dir, 'rb') as compressed:
        decomp = zstandard.ZstdDecompressor()
        with open(dir[:-4], 'wb') as destination:
            decomp.copy_stream(compressed, destination)

    print("now starting to process the data...")
    newjson = open(outputdir, 'w')
    data = loadJsonL(dir[:-4])
    for jsonline in data:
        # extract the filename extension of the current line
        # which is under meta > file_name
        ext = jsonline['meta']['file_name']
        # ignore the filename and only get the extension
        if(ext != ''):
            ext = ext.split('.')[-1].lower()
        # if extension is py, then process the line
        # processing means creating a file with name as this line number

        if ext == 'py':
            newjson.write(json.dumps(jsonline))
            newjson.write('\n')
    print("done!")
    newjson.close()
    os.remove(dir)
    os.remove(dir[:-4])


urls = get_all_website_links(
    "https://the-eye.eu/public/AI/training_data/code_clippy_data/code_clippy_dedup_data/train/")
for i in urls:
    if i.find("_default.jsonl.zst"):
        try:
            processZSTLink(i, './zsts/', './pyjsons/')
        except:
            print("Error happened")
