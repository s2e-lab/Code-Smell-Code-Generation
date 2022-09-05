def get_upload_url(credentials):
    '''
    Returns upload URL using new upload API
    '''
    request_url = "https://a.mapillary.com/v3/users/{}/upload_secrets?client_id={}".format(
        credentials["MAPSettingsUserKey"], CLIENT_ID)
    request = urllib2.Request(request_url)
    request.add_header('Authorization', 'Bearer {}'.format(
        credentials["user_upload_token"]))
    try:
        response = json.loads(urllib2.urlopen(request).read())
    except requests.exceptions.HTTPError as e:
        print("Error getting upload parameters, upload could not start")
        sys.exit(1)
    return response