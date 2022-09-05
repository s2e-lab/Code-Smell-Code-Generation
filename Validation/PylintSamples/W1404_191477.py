def movielens100k(data_set='movielens100k'):
    """Data set of movie ratings collected by the University of Minnesota and 'cleaned up' for use."""
    if not data_available(data_set):
        import zipfile
        download_data(data_set)
        dir_path = os.path.join(data_path, data_set)
        zip = zipfile.ZipFile(os.path.join(dir_path, 'ml-100k.zip'), 'r')
        for name in zip.namelist():
            zip.extract(name, dir_path)
    import pandas as pd
    encoding = 'latin-1'
    movie_path = os.path.join(data_path, 'movielens100k', 'ml-100k')
    items = pd.read_csv(os.path.join(movie_path, 'u.item'), index_col = 'index', header=None, sep='|',names=['index', 'title', 'date', 'empty', 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation', 'Children''s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding=encoding)
    users = pd.read_csv(os.path.join(movie_path, 'u.user'), index_col = 'index', header=None, sep='|', names=['index', 'age', 'sex', 'job', 'id'], encoding=encoding)
    parts = ['u1.base', 'u1.test', 'u2.base', 'u2.test','u3.base', 'u3.test','u4.base', 'u4.test','u5.base', 'u5.test','ua.base', 'ua.test','ub.base', 'ub.test']
    ratings = []
    for part in parts:
        rate_part = pd.read_csv(os.path.join(movie_path, part), index_col = 'index', header=None, sep='\t', names=['user', 'item', 'rating', 'index'], encoding=encoding)
        rate_part['split'] = part
        ratings.append(rate_part)
    Y = pd.concat(ratings)
    return data_details_return({'Y':Y, 'film_info':items, 'user_info':users, 'info': 'The Movielens 100k data'}, data_set)