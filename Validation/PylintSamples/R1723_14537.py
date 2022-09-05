def _get_csv_fieldnames(csv_reader):
    """Finds fieldnames in Polarion exported csv file."""
    fieldnames = []
    for row in csv_reader:
        for col in row:
            field = (
                col.strip()
                .replace('"', "")
                .replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .lower()
            )
            fieldnames.append(field)
        if "id" in fieldnames:
            break
        else:
            # this is not a row with fieldnames
            del fieldnames[:]
    if not fieldnames:
        return None
    # remove trailing unannotated fields
    while True:
        field = fieldnames.pop()
        if field:
            fieldnames.append(field)
            break
    # name unannotated fields
    suffix = 1
    for index, field in enumerate(fieldnames):
        if not field:
            fieldnames[index] = "field{}".format(suffix)
            suffix += 1

    return fieldnames