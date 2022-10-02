def load_model(model_file):
    """Load a model by its file. This includes the model itself, but also
       the preprocessing queue, the feature list and the output semantics.
    """
    # Extract tar
    with tarfile.open(model_file) as tar:
        tarfolder = tempfile.mkdtemp()
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner) 
            
        
        safe_extract(tar, path=tarfolder)

    from . import features
    from . import preprocessing

    # Get the preprocessing
    with open(os.path.join(tarfolder, "preprocessing.yml"), 'r') as ymlfile:
        preprocessing_description = yaml.load(ymlfile)
    preprocessing_queue = preprocessing.get_preprocessing_queue(
        preprocessing_description['queue'])

    # Get the features
    with open(os.path.join(tarfolder, "features.yml"), 'r') as ymlfile:
        feature_description = yaml.load(ymlfile)
    feature_str_list = feature_description['features']
    feature_list = features.get_features(feature_str_list)

    # Get the model
    import nntoolkit.utils
    model = nntoolkit.utils.get_model(model_file)

    output_semantics_file = os.path.join(tarfolder, 'output_semantics.csv')
    output_semantics = nntoolkit.utils.get_outputs(output_semantics_file)

    # Cleanup
    shutil.rmtree(tarfolder)

    return (preprocessing_queue, feature_list, model, output_semantics)