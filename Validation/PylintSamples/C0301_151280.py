def update(self, account_sid=values.unset, api_version=values.unset,
               friendly_name=values.unset, sms_application_sid=values.unset,
               sms_fallback_method=values.unset, sms_fallback_url=values.unset,
               sms_method=values.unset, sms_url=values.unset,
               status_callback=values.unset, status_callback_method=values.unset,
               voice_application_sid=values.unset,
               voice_caller_id_lookup=values.unset,
               voice_fallback_method=values.unset, voice_fallback_url=values.unset,
               voice_method=values.unset, voice_url=values.unset,
               emergency_status=values.unset, emergency_address_sid=values.unset,
               trunk_sid=values.unset, voice_receive_mode=values.unset,
               identity_sid=values.unset, address_sid=values.unset):
        """
        Update the IncomingPhoneNumberInstance

        :param unicode account_sid: The SID of the Account that created the resource to update
        :param unicode api_version: The API version to use for incoming calls made to the phone number
        :param unicode friendly_name: A string to describe the resource
        :param unicode sms_application_sid: Unique string that identifies the application
        :param unicode sms_fallback_method: HTTP method used with sms_fallback_url
        :param unicode sms_fallback_url: The URL we call when an error occurs while executing TwiML
        :param unicode sms_method: The HTTP method to use with sms_url
        :param unicode sms_url: The URL we should call when the phone number receives an incoming SMS message
        :param unicode status_callback: The URL we should call to send status information to your application
        :param unicode status_callback_method: The HTTP method we should use to call status_callback
        :param unicode voice_application_sid: The SID of the application to handle the phone number
        :param bool voice_caller_id_lookup: Whether to lookup the caller's name
        :param unicode voice_fallback_method: The HTTP method used with fallback_url
        :param unicode voice_fallback_url: The URL we will call when an error occurs in TwiML
        :param unicode voice_method: The HTTP method used with the voice_url
        :param unicode voice_url: The URL we should call when the phone number receives a call
        :param IncomingPhoneNumberInstance.EmergencyStatus emergency_status: Whether the phone number is enabled for emergency calling
        :param unicode emergency_address_sid: The emergency address configuration to use for emergency calling
        :param unicode trunk_sid: SID of the trunk to handle phone calls to the phone number
        :param IncomingPhoneNumberInstance.VoiceReceiveMode voice_receive_mode: Incoming call type: fax or voice
        :param unicode identity_sid: Unique string that identifies the identity associated with number
        :param unicode address_sid: The SID of the Address resource associated with the phone number

        :returns: Updated IncomingPhoneNumberInstance
        :rtype: twilio.rest.api.v2010.account.incoming_phone_number.IncomingPhoneNumberInstance
        """
        return self._proxy.update(
            account_sid=account_sid,
            api_version=api_version,
            friendly_name=friendly_name,
            sms_application_sid=sms_application_sid,
            sms_fallback_method=sms_fallback_method,
            sms_fallback_url=sms_fallback_url,
            sms_method=sms_method,
            sms_url=sms_url,
            status_callback=status_callback,
            status_callback_method=status_callback_method,
            voice_application_sid=voice_application_sid,
            voice_caller_id_lookup=voice_caller_id_lookup,
            voice_fallback_method=voice_fallback_method,
            voice_fallback_url=voice_fallback_url,
            voice_method=voice_method,
            voice_url=voice_url,
            emergency_status=emergency_status,
            emergency_address_sid=emergency_address_sid,
            trunk_sid=trunk_sid,
            voice_receive_mode=voice_receive_mode,
            identity_sid=identity_sid,
            address_sid=address_sid,
        )