def list_from_json(source_list_json):
    """
    Deserialise all the items in source_list from json
    """
    result = []
    if source_list_json == [] or source_list_json == None:
        return result
    for list_item in source_list_json:
        item = json.loads(list_item)
        try:
            if item['class_name'] == 'Departure':
                temp = Departure()
            elif item['class_name'] == 'Disruption':
                temp = Disruption()
            elif item['class_name'] == 'Station':
                temp = Station()
            elif item['class_name'] == 'Trip':
                temp = Trip()
            elif item['class_name'] == 'TripRemark':
                temp = TripRemark()
            elif item['class_name'] == 'TripStop':
                temp = TripStop()
            elif item['class_name'] == 'TripSubpart':
                temp = TripSubpart()
            else:
                print('Unrecognised Class ' + item['class_name'] + ', skipping')
                continue
            temp.from_json(list_item)
            result.append(temp)
        except KeyError:
            print('Unrecognised item with no class_name, skipping')
            continue
    return result