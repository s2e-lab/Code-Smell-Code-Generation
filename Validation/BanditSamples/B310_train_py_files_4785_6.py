import urllib.request, re

def get_member_since(user):
    URL = "https://www.codewars.com/users/"
    page = urllib.request.urlopen(URL + user)
    join_date = re.search("[A-Z][a-z]+ d+", str(page.read()))
    return join_date.group()
