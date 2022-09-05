def _get_rows(self):
        """
        Return all rows on page
        """
        html = requests.get(self.url.build()).text
        if re.search('did not match any documents', html):
            return []
        pq = PyQuery(html)
        rows = pq("table.data").find("tr")
        return map(rows.eq, range(rows.size()))[1:]