def make_bucket_policy_statements(bucket):
    """Return the statemtns in a bucket policy as a dict of dicts"""
    import yaml
    from os.path import dirname, join, abspath
    import copy
    import metatab

    with open(join(dirname(abspath(metatab.__file__)), 'support', 'policy_parts.yaml')) as f:
        parts = yaml.load(f)

    statements = {}

    cl = copy.deepcopy(parts['list'])
    cl['Resource'] = arn_prefix + bucket
    statements['list'] = cl

    cl = copy.deepcopy(parts['bucket'])
    cl['Resource'] = arn_prefix + bucket
    statements['bucket'] = cl

    for sd in TOP_LEVEL_DIRS:
        cl = copy.deepcopy(parts['read'])
        cl['Resource'] = arn_prefix + bucket + '/' + sd + '/*'
        cl['Sid'] = cl['Sid'].title() + sd.title()

        statements[cl['Sid']] = cl

        cl = copy.deepcopy(parts['write'])
        cl['Resource'] = arn_prefix + bucket + '/' + sd + '/*'
        cl['Sid'] = cl['Sid'].title() + sd.title()

        statements[cl['Sid']] = cl

        cl = copy.deepcopy(parts['listb'])
        cl['Resource'] = arn_prefix + bucket
        cl['Sid'] = cl['Sid'].title() + sd.title()
        cl['Condition']['StringLike']['s3:prefix'] = [sd + '/*']

        statements[cl['Sid']] = cl

    return statements