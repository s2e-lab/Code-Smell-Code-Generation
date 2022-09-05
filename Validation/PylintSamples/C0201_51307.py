def get_overlay(self):
        """
        Function make 3D data from dicom file slices. There are usualy
        more overlays in the data.
        """
        overlay = {}
        dcmlist = self.files_in_serie

        for i in range(len(dcmlist)):
            onefile = dcmlist[i]
            logger.info("reading '%s'" % onefile)
            data = self._read_file(onefile)

            if len(overlay) == 0:
                # first there is created dictionary with
                # avalible overlay indexes
                for i_overlay in range(0, 50):
                    try:
                        # overlay index
                        data2d = decode_overlay_slice(data, i_overlay)
                        # mport pdb; pdb.set_trace()
                        shp2 = data2d.shape
                        overlay[i_overlay] = np.zeros([len(dcmlist), shp2[0],
                                                       shp2[1]], dtype=np.int8)
                        overlay[i_overlay][-i - 1, :, :] = data2d

                    except Exception:
                        # exception is exceptetd. We are trying numbers 0-50
                        # logger.exception('Problem with overlay image number ' +
                        #               str(i_overlay))
                        pass

            else:
                for i_overlay in overlay.keys():
                    try:
                        data2d = decode_overlay_slice(data, i_overlay)
                        overlay[i_overlay][-i - 1, :, :] = data2d
                    except Exception:
                        logger.warning('Problem with overlay number ' +
                                       str(i_overlay))

        return overlay