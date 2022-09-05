def _resolved_url(self):
        """Return a URL that properly combines the base_url and a possibly relative
        resource url"""

        if not self.url:
            return None

        u = parse_app_url(self.url)

        if u.scheme == 'index':
            u = u.resolve()

        if u.scheme != 'file':
            # Hopefully means the URL is http, https, ftp, etc.

            return u
        elif u.resource_format == 'ipynb':

            # This shouldn't be a special case, but ...
            t = self.doc.package_url.inner.join_dir(self.url)
            t = t.as_type(type(u))
            t.fragment = u.fragment

            return t

        elif u.proto == 'metatab':

            u = self.expanded_url

            return u.get_resource().get_target()

        elif u.proto == 'metapack':

            u = self.expanded_url

            if u.resource:
                return u.resource.resolved_url.get_resource().get_target()
            else:
                return u

        if u.scheme == 'file':

            return self.expanded_url

        elif False:


            assert isinstance(self.doc.package_url, MetapackPackageUrl), (type(self.doc.package_url), self.doc.package_url)

            try:
                t = self.doc.package_url.resolve_url(self.url) # Why are we doing this?

                # Also a hack
                t.scheme_extension = parse_app_url(self.url).scheme_extension

                # Another Hack!
                try:
                    if not any(t.fragment) and any(u.fragment):
                        t.fragment = u.fragment
                except TypeError:
                    if not t.fragment and u.fragment:
                        t.fragment = u.fragment


                # Yet more hack!
                t = parse_app_url(str(t))

                return t

            except ResourceError as e:
                # This case happens when a filesystem packages has a non-standard metadata name
                # Total hack
                raise

        else:
            raise ResourceError('Unknown case for url {} '.format(self.url))