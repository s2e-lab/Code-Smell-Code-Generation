def prop_triangle(self, **kwargs):
        """
        Makes corner plot of only observable properties.

        The idea here is to compare the predictions of the samples
        with the actual observed data---this can be a quick way to check
        if there are outlier properties that aren't predicted well
        by the model.

        :param **kwargs:
            Keyword arguments passed to :func:`StarModel.triangle`.

        :return:
            Figure object containing corner plot.
         
        """
        truths = []
        params = []
        for p in self.properties:
            try:
                val, err = self.properties[p]
            except:
                continue

            if p in self.ic.bands:
                params.append('{}_mag'.format(p))
                truths.append(val)
            elif p=='parallax':
                params.append('distance')
                truths.append(1/(val/1000.))
            else:
                params.append(p)
                truths.append(val)
        return self.triangle(params, truths=truths, **kwargs)