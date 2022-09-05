def from_element(cls, element):     # pylint: disable=R0914
        """Create an instance of this class from an ElementTree xml Element.

        An alternative constructor. The element must be a DIDL-Lite <item> or
        <container> element, and must be properly namespaced.

        Args:
            xml (~xml.etree.ElementTree.Element): An
                :class:`~xml.etree.ElementTree.Element` object.
        """
        # We used to check here that we have the right sort of element,
        # ie a container or an item. But Sonos seems to use both
        # indiscriminately, eg a playlistContainer can be an item or a
        # container. So we now just check that it is one or the other.
        tag = element.tag
        if not (tag.endswith('item') or tag.endswith('container')):
            raise DIDLMetadataError(
                "Wrong element. Expected <item> or <container>,"
                " got <{0}> for class {1}'".format(
                    tag, cls.item_class))
        # and that the upnp matches what we are expecting
        item_class = element.find(ns_tag('upnp', 'class')).text

        # In case this class has an # specified unofficial
        # subclass, ignore it by stripping it from item_class
        if '.#' in item_class:
            item_class = item_class[:item_class.find('.#')]

        if item_class != cls.item_class:
            raise DIDLMetadataError(
                "UPnP class is incorrect. Expected '{0}',"
                " got '{1}'".format(cls.item_class, item_class))

        # parent_id, item_id  and restricted are stored as attributes on the
        # element
        item_id = element.get('id', None)
        if item_id is None:
            raise DIDLMetadataError("Missing id attribute")
        item_id = really_unicode(item_id)
        parent_id = element.get('parentID', None)
        if parent_id is None:
            raise DIDLMetadataError("Missing parentID attribute")
        parent_id = really_unicode(parent_id)

        # CAUTION: This implementation deviates from the spec.
        # Elements are normally required to have a `restricted` tag, but
        # Spotify Direct violates this. To make it work, a missing restricted
        # tag is interpreted as `restricted = True`.
        restricted = element.get('restricted', None)
        restricted = False if restricted in [0, 'false', 'False'] else True

        # Similarily, all elements should have a title tag, but Spotify Direct
        # does not comply
        title_elt = element.find(ns_tag('dc', 'title'))
        if title_elt is None or not title_elt.text:
            title = ''
        else:
            title = really_unicode(title_elt.text)

        # Deal with any resource elements
        resources = []
        for res_elt in element.findall(ns_tag('', 'res')):
            resources.append(
                DidlResource.from_element(res_elt))

        # and the desc element (There is only one in Sonos)
        desc = element.findtext(ns_tag('', 'desc'))

        # Get values of the elements listed in _translation and add them to
        # the content dict
        content = {}
        for key, value in cls._translation.items():
            result = element.findtext(ns_tag(*value))
            if result is not None:
                # We store info as unicode internally.
                content[key] = really_unicode(result)

        # Convert type for original track number
        if content.get('original_track_number') is not None:
            content['original_track_number'] = \
                int(content['original_track_number'])

        # Now pass the content dict we have just built to the main
        # constructor, as kwargs, to create the object
        return cls(title=title, parent_id=parent_id, item_id=item_id,
                   restricted=restricted, resources=resources, desc=desc,
                   **content)