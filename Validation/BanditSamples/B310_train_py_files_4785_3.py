from bs4 import BeautifulSoup
from urllib.request import urlopen

def get_member_since(username):
    return BeautifulSoup(urlopen("https://www.codewars.com/users/" + username)).find(string="Member Since:").next_element
