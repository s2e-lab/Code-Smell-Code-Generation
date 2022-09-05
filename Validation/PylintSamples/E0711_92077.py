def put_stream(self, rel_path, metadata=None, cb=None):
        """return a file object to write into the cache. The caller
        is responsibile for closing the stream. Bad things happen
        if you dont close the stream
        """

        class flo:

            def __init__(self, this, sink, upstream, repo_path):
                self.this = this
                self.sink = sink
                self.upstream = upstream
                self.repo_path = repo_path

            @property
            def repo_path(self):
                return self.repo_path

            def write(self, d):
                self.sink.write(d)

                if self.upstream:
                    self.upstream.write(d)

            def writelines(self, lines):
                raise NotImplemented()

            def close(self):
                self.sink.close()

                size = os.path.getsize(self.repo_path)

                self.this.add_record(rel_path, size)
                self.this._free_up_space(size, this_rel_path=rel_path)

                if self.upstream:
                    self.upstream.close()
                    
            def __enter__(self): # Can be used as a context!
                return self

            def __exit__(self, type_, value, traceback):
                if type_:
                    return False

        if not isinstance(rel_path, basestring):
            rel_path = rel_path.cache_key

        repo_path = os.path.join(self.cache_dir, rel_path.strip('/'))

        if not os.path.isdir(os.path.dirname(repo_path)):
            os.makedirs(os.path.dirname(repo_path))

        self.put_metadata(rel_path, metadata=metadata)

        sink = open(repo_path, 'w+')
        upstream = self.upstream.put_stream(
            rel_path,
            metadata=metadata) if self.upstream else None

        return flo(self, sink, upstream, repo_path)