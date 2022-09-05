def _adb_screencap(self, scale=1.0):
        """
        capture screen with adb shell screencap
        """
        remote_file = tempfile.mktemp(dir='/data/local/tmp/', prefix='screencap-', suffix='.png')
        local_file = tempfile.mktemp(prefix='atx-screencap-', suffix='.png')
        self.shell('screencap', '-p', remote_file)
        try:
            self.pull(remote_file, local_file)
            image = imutils.open_as_pillow(local_file)
            if scale is not None and scale != 1.0:
                image = image.resize([int(scale * s) for s in image.size], Image.BICUBIC)
            rotation = self.rotation()
            if rotation:
                method = getattr(Image, 'ROTATE_{}'.format(rotation*90))
                image = image.transpose(method)
            return image
        finally:
            self.remove(remote_file)
            os.unlink(local_file)