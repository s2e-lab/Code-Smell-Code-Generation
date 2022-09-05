def pmlr(volumes='all', data_set='pmlr'):
    """Abstracts from the Proceedings of Machine Learning Research"""
    if not data_available(data_set):
        download_data(data_set)
        
    proceedings_file = open(os.path.join(data_path, data_set, 'proceedings.yaml'), 'r')
    import yaml
    proceedings = yaml.load(proceedings_file)
    
    # Create a new resources entry for downloading contents of proceedings.
    data_name_full = 'pmlr_volumes'
    data_resources[data_name_full] = data_resources[data_set].copy()
    data_resources[data_name_full]['files'] = []
    data_resources[data_name_full]['dirs'] = []
    data_resources[data_name_full]['urls'] = []
    for entry in proceedings:
        if volumes=='all' or entry['volume'] in volumes:
            file = entry['yaml'].split('/')[-1]
            dir = 'v' + str(entry['volume'])
            data_resources[data_name_full]['files'].append([file])
            data_resources[data_name_full]['dirs'].append([dir])
            data_resources[data_name_full]['urls'].append(data_resources[data_set]['urls'][0])
    Y = []
    # Download the volume data
    if not data_available(data_name_full):
        download_data(data_name_full)
    for entry in reversed(proceedings):
        volume =  entry['volume']
        if volumes == 'all' or volume in volumes:
            file = entry['yaml'].split('/')[-1]
            volume_file = open(os.path.join(
                data_path, data_name_full,
                'v'+str(volume), file
                ), 'r')
            Y+=yaml.load(volume_file)
    if pandas_available:
        Y = pd.DataFrame(Y)
        Y['published'] = pd.to_datetime(Y['published'])
        #Y.columns.values[4] = json_object('authors')
        #Y.columns.values[7] = json_object('editors')
        Y['issued'] = Y['issued'].apply(lambda x: np.datetime64(datetime.datetime(*x['date-parts'])))
        Y['author'] = Y['author'].apply(lambda x: [str(author['given']) + ' ' + str(author['family']) for author in x])
        Y['editor'] = Y['editor'].apply(lambda x: [str(editor['given']) + ' ' + str(editor['family']) for editor in x])
        columns = list(Y.columns)
        columns[14] = datetime64_('published')
        columns[11] = datetime64_('issued')
        Y.columns = columns
        
    return data_details_return({'Y' : Y, 'info' : 'Data is a pandas data frame containing each paper, its abstract, authors, volumes and venue.'}, data_set)