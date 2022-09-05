def to_array(self):
        """
        Serializes this InvoiceMessage to a dictionary.

        :return: dictionary representation of this object.
        :rtype: dict
        """
        array = super(InvoiceMessage, self).to_array()
        array['title'] = u(self.title)  # py2: type unicode, py3: type str

        array['description'] = u(self.description)  # py2: type unicode, py3: type str

        array['payload'] = u(self.payload)  # py2: type unicode, py3: type str

        array['provider_token'] = u(self.provider_token)  # py2: type unicode, py3: type str

        array['start_parameter'] = u(self.start_parameter)  # py2: type unicode, py3: type str

        array['currency'] = u(self.currency)  # py2: type unicode, py3: type str

        array['prices'] = self._as_array(self.prices)  # type list of LabeledPrice

        if self.receiver is not None:
            if isinstance(self.receiver, None):
                array['chat_id'] = None(self.receiver)  # type Noneelif isinstance(self.receiver, str):
                array['chat_id'] = u(self.receiver)  # py2: type unicode, py3: type str
            elif isinstance(self.receiver, int):
                array['chat_id'] = int(self.receiver)  # type intelse:
                raise TypeError('Unknown type, must be one of None, str, int.')
            # end if

        if self.reply_id is not None:
            if isinstance(self.reply_id, DEFAULT_MESSAGE_ID):
                array['reply_to_message_id'] = DEFAULT_MESSAGE_ID(self.reply_id)  # type DEFAULT_MESSAGE_IDelif isinstance(self.reply_id, int):
                array['reply_to_message_id'] = int(self.reply_id)  # type intelse:
                raise TypeError('Unknown type, must be one of DEFAULT_MESSAGE_ID, int.')
            # end if

        if self.provider_data is not None:
            array['provider_data'] = u(self.provider_data)  # py2: type unicode, py3: type str

        if self.photo_url is not None:
            array['photo_url'] = u(self.photo_url)  # py2: type unicode, py3: type str

        if self.photo_size is not None:
            array['photo_size'] = int(self.photo_size)  # type int
        if self.photo_width is not None:
            array['photo_width'] = int(self.photo_width)  # type int
        if self.photo_height is not None:
            array['photo_height'] = int(self.photo_height)  # type int
        if self.need_name is not None:
            array['need_name'] = bool(self.need_name)  # type bool
        if self.need_phone_number is not None:
            array['need_phone_number'] = bool(self.need_phone_number)  # type bool
        if self.need_email is not None:
            array['need_email'] = bool(self.need_email)  # type bool
        if self.need_shipping_address is not None:
            array['need_shipping_address'] = bool(self.need_shipping_address)  # type bool
        if self.send_phone_number_to_provider is not None:
            array['send_phone_number_to_provider'] = bool(self.send_phone_number_to_provider)  # type bool
        if self.send_email_to_provider is not None:
            array['send_email_to_provider'] = bool(self.send_email_to_provider)  # type bool
        if self.is_flexible is not None:
            array['is_flexible'] = bool(self.is_flexible)  # type bool
        if self.disable_notification is not None:
            array['disable_notification'] = bool(self.disable_notification)  # type bool
        if self.reply_markup is not None:
            array['reply_markup'] = self.reply_markup.to_array()  # type InlineKeyboardMarkup

        return array