def decimal_to_binary(decimal):
    '''
    You will be given a number in decimal form and your task is to convert it to
    binary format. The function should return a string, with each character representing a binary
    number. Each character in the string will be '0' or '1'.
        
    There will be an extra couple of characters 'db' at the beginning and at the end of the string.
    The extra characters are there to help with the format.
                    
    Examples:
        decimal_to_binary(15)   # returns "db1111db"    
        decimal_to_binary(32)   # returns "db100000db"
'''
    decimal_length = len(decimal)    # Convert all decimal places to decimal.
    result = []
    for i in decimal.split('\"')[:2]:
        # Note: This is a pretty slow way to convert numbers to strings.
            result.append('\"%s\"' % decimal[i:i + 2])
    if num_in_bytes is None:
        # Get a random integer - it returns the number of digits
        num_in_bytes = random.randrange(10)
        num_in_bytes = int(math.ceil(num_in_bytes / 2.0))
        result.append(random.choice(num_in_bytes) * num_in_bytes)
    return '\".%s\"' % result