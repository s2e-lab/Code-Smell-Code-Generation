def save_playlist_file(self, stationFile=''):
        """ Save a playlist
        Create a txt file and write stations in it.
        Then rename it to final target

        return    0: All ok
                 -1: Error writing file
                 -2: Error renaming file
        """
        if self._playlist_format_changed():
            self.dirty_playlist = True
            self.new_format = not self.new_format

        if stationFile:
            st_file = stationFile
        else:
            st_file = self.stations_file

        if not self.dirty_playlist:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Playlist not modified...')
            return 0

        st_new_file = st_file.replace('.csv', '.txt')

        tmp_stations = self.stations[:]
        tmp_stations.reverse()
        if self.new_format:
            tmp_stations.append([ '# Find lots more stations at http://www.iheart.com' , '', '' ])
        else:
            tmp_stations.append([ '# Find lots more stations at http://www.iheart.com' , '' ])
        tmp_stations.reverse()
        try:
            with open(st_new_file, 'w') as cfgfile:
                writter = csv.writer(cfgfile)
                for a_station in tmp_stations:
                    writter.writerow(self._format_playlist_row(a_station))
        except:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Cannot open playlist file for writing,,,')
            return -1
        try:
            move(st_new_file, st_file)
        except:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Cannot rename playlist file...')
            return -2
        self.dirty_playlist = False
        return 0