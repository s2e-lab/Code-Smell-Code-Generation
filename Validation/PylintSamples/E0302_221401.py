def stream_file(self, project, path):
        """
        Read file of a project and stream it

        :param project: A project object
        :param path: The path of the file in the project
        :returns: A file stream
        """

        # Due to Python 3.4 limitation we can't use with and asyncio
        # https://www.python.org/dev/peps/pep-0492/
        # that why we wrap the answer
        class StreamResponse:

            def __init__(self, response):
                self._response = response

            def __enter__(self):
                return self._response.content

            def __exit__(self):
                self._response.close()

        url = self._getUrl("/projects/{}/stream/{}".format(project.id, path))
        response = yield from self._session().request("GET", url, auth=self._auth, timeout=None)
        if response.status == 404:
            raise aiohttp.web.HTTPNotFound(text="{} not found on compute".format(path))
        elif response.status == 403:
            raise aiohttp.web.HTTPForbidden(text="forbidden to open {} on compute".format(path))
        elif response.status != 200:
            raise aiohttp.web.HTTPInternalServerError(text="Unexpected error {}: {}: while opening {} on compute".format(response.status,
                                                                                                                         response.reason,
                                                                                                                         path))
        return StreamResponse(response)