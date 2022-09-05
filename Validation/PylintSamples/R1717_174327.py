def describe(name, tags=None, region=None, key=None, keyid=None,
             profile=None):
    '''
    Return RDS instance details.

    CLI example::

        salt myminion boto_rds.describe myrds

    '''
    res = __salt__['boto_rds.exists'](name, tags, region, key, keyid,
                                      profile)
    if not res.get('exists'):
        return {'exists': bool(res), 'message':
                'RDS instance {0} does not exist.'.format(name)}

    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        if not conn:
            return {'results': bool(conn)}

        rds = conn.describe_db_instances(DBInstanceIdentifier=name)
        rds = [
            i for i in rds.get('DBInstances', [])
            if i.get('DBInstanceIdentifier') == name
        ].pop(0)

        if rds:
            keys = ('DBInstanceIdentifier', 'DBInstanceClass', 'Engine',
                    'DBInstanceStatus', 'DBName', 'AllocatedStorage',
                    'PreferredBackupWindow', 'BackupRetentionPeriod',
                    'AvailabilityZone', 'PreferredMaintenanceWindow',
                    'LatestRestorableTime', 'EngineVersion',
                    'AutoMinorVersionUpgrade', 'LicenseModel',
                    'Iops', 'CharacterSetName', 'PubliclyAccessible',
                    'StorageType', 'TdeCredentialArn', 'DBInstancePort',
                    'DBClusterIdentifier', 'StorageEncrypted', 'KmsKeyId',
                    'DbiResourceId', 'CACertificateIdentifier',
                    'CopyTagsToSnapshot', 'MonitoringInterval',
                    'MonitoringRoleArn', 'PromotionTier',
                    'DomainMemberships')
            return {'rds': dict([(k, rds.get(k)) for k in keys])}
        else:
            return {'rds': None}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}
    except IndexError:
        return {'rds': None}