from subprocess import check_output

def get_output(s):
    return check_output(s.split()).decode('utf-8') 
