def ordering_beers(beers):
    assert 0 <= beers < 100
    
    units = ["", "jeden", "dwa", "trzy", "cztery", "piec", "szesc" , "siedem", "osiem", "dziewiec",
            "dziesiec", "jedenascie", "dwanascie", "trzynascie", "czternascie", "pietnascie", "szesnascie", "siedemnascie", "osiemnascie", "dziewietnascie"]
    tens  = ["dwadziescia", "trzydziesci", "czterdziesci", "piecdziesiat", "szescdziesiat", "siedemdziesiat", "osiemdziesiat", "dziewiecdziesiat"]
    
    order = ["Woda mineralna", "Jedno piwo"]
    
    for i, num in enumerate(units[2:], 2):
        order.append("%s piw%s" % (num, "a" if i in [2, 3, 4] else ""))
    
    for n in tens:
        for i in range(10):
            order.append("%s%s%s piw%s" % (n, " " * bool(i), units[i], "a" if i in [2, 3, 4] else ""))
    
    return order[beers].capitalize() + " poprosze"
