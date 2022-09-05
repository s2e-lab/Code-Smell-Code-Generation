def _couple_nic(self, userid, vdev, vswitch_name,
                    active=False):
        """Couple NIC to vswitch by adding vswitch into user direct."""
        if active:
            self._is_active(userid)

        msg = ('Start to couple nic device %(vdev)s of guest %(vm)s '
               'with vswitch %(vsw)s'
                % {'vdev': vdev, 'vm': userid, 'vsw': vswitch_name})
        LOG.info(msg)
        requestData = ' '.join((
            'SMAPI %s' % userid,
            "API Virtual_Network_Adapter_Connect_Vswitch_DM",
            "--operands",
            "-v %s" % vdev,
            "-n %s" % vswitch_name))

        try:
            self._request(requestData)
        except exception.SDKSMTRequestFailed as err:
            LOG.error("Failed to couple nic %s to vswitch %s for user %s "
                      "in the guest's user direct, error: %s" %
                      (vdev, vswitch_name, userid, err.format_message()))
            self._couple_inactive_exception(err, userid, vdev, vswitch_name)

        # the inst must be active, or this call will failed
        if active:
            requestData = ' '.join((
                'SMAPI %s' % userid,
                'API Virtual_Network_Adapter_Connect_Vswitch',
                "--operands",
                "-v %s" % vdev,
                "-n %s" % vswitch_name))

            try:
                self._request(requestData)
            except (exception.SDKSMTRequestFailed,
                    exception.SDKInternalError) as err1:
                results1 = err1.results
                msg1 = err1.format_message()
                if ((results1 is not None) and
                    (results1['rc'] == 204) and
                    (results1['rs'] == 20)):
                    LOG.warning("Virtual device %s already connected "
                                "on the active guest system", vdev)
                else:
                    persist_OK = True
                    requestData = ' '.join((
                        'SMAPI %s' % userid,
                        'API Virtual_Network_Adapter_Disconnect_DM',
                        "--operands",
                        '-v %s' % vdev))
                    try:
                        self._request(requestData)
                    except (exception.SDKSMTRequestFailed,
                            exception.SDKInternalError) as err2:
                        results2 = err2.results
                        msg2 = err2.format_message()
                        if ((results2 is not None) and
                            (results2['rc'] == 212) and
                            (results2['rs'] == 32)):
                            persist_OK = True
                        else:
                            persist_OK = False
                    if persist_OK:
                        self._couple_active_exception(err1, userid, vdev,
                                                      vswitch_name)
                    else:
                        raise exception.SDKNetworkOperationError(rs=3,
                                    nic=vdev, vswitch=vswitch_name,
                                    couple_err=msg1, revoke_err=msg2)

        """Update information in switch table."""
        self._NetDbOperator.switch_update_record_with_switch(userid, vdev,
                                                             vswitch_name)
        msg = ('Couple nic device %(vdev)s of guest %(vm)s '
               'with vswitch %(vsw)s successfully'
                % {'vdev': vdev, 'vm': userid, 'vsw': vswitch_name})
        LOG.info(msg)