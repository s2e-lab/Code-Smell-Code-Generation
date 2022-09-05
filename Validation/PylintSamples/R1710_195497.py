def _get_service_cluster_desired_count(template, parameters, service_name):
    """Helper to extract the configured desiredCount attribute from the
    template.

    :param template: cloudformation template (json)
    :param parameters: list of {'ParameterKey': 'x1', 'ParameterValue': 'y1'}
    :param service_name: logical resource name of the ECS service
    :return: cluster, desiredCount
    """
    params = {e['ParameterKey']: e['ParameterValue'] for e in parameters}
    service = template.get('Resources', {}).get(service_name, None)
    if service:
        assert service['Type'] == 'AWS::ECS::Service'
        cluster = service.get('Properties', {}).get('Cluster', None)
        desired_count = service.get('Properties', {}).get('DesiredCount', None)
        if 'Ref' in cluster:
            cluster = params.get(cluster['Ref'], None)
        if not isinstance(desired_count, int) and 'Ref' in desired_count:
            desired_count = params.get(desired_count['Ref'], None)
        return cluster, int(desired_count)