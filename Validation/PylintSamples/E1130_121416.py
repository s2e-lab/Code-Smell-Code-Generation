def diag(image, mask=None, iterations=1):
    '''4-connect pixels that are 8-connected
    
    0 0 0     0 0 ?
    0 0 1 ->  0 1 1
    0 1 0     ? 1 ?
    
    '''
    global diag_table
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = table_lookup(masked_image, diag_table, False, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result