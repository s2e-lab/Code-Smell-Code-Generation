def render_in_browser(self, **kwargs):
        """Render the graph, open it in your browser with black magic"""
        try:
            from lxml.html import open_in_browser
        except ImportError:
            raise ImportError('You must install lxml to use render in browser')
        kwargs.setdefault('force_uri_protocol', 'https')
        open_in_browser(self.render_tree(**kwargs), encoding='utf-8')