def original_number(s):
    s_list = list(s)
    numbers = [(0, 'ZERO'), (2, 'TWO'), (6, 'SIX'), (4, 'FOUR'),  (1, 'ONE'), (5, 'FIVE'), (7, 'SEVEN'), (9, 'NINE'), (3, 'THREE'), (8, 'EIGHT')]  
    secret_number = ''
    for i, number in numbers:
        while all([c in s_list for c in number]):            
            [s_list.remove(c) for c in number]  
            secret_number += str(i)     

    return ''.join(sorted(secret_number))
