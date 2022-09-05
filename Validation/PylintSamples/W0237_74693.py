def render_path(self) -> str:
        """Render path by filling the path template with video information."""
        # TODO: Fix defaults when date is not found (empty string or None)
        # https://stackoverflow.com/questions/23407295/default-kwarg-values-for-pythons-str-format-method

        from string import Formatter

        class UnseenFormatter(Formatter):
            def get_value(self, key, args, kwds):
                if isinstance(key, str):
                    try:
                        return kwds[key]
                    except KeyError:
                        return key
                else:
                    return super().get_value(key, args, kwds)

        data = self.video.data
        site_name = data['site']

        try:
            template = self.templates[site_name]
        except KeyError:
            raise NoTemplateFoundError

        fmt = UnseenFormatter()
        filename_raw = fmt.format(template, **data)
        filename = clean_filename(filename_raw)
        path = os.path.join(self.download_dir, filename)
        return path