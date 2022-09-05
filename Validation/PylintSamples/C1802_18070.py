def aggregate_region(self, variable, region='World', subregions=None,
                         components=None, append=False):
        """Compute the aggregate of timeseries over a number of regions
        including variable components only defined at the `region` level

        Parameters
        ----------
        variable: str
            variable for which the aggregate should be computed
        region: str, default 'World'
            dimension
        subregions: list of str
            list of subregions, defaults to all regions other than `region`
        components: list of str
            list of variables, defaults to all sub-categories of `variable`
            included in `region` but not in any of `subregions`
        append: bool, default False
            append the aggregate timeseries to `data` and return None,
            else return aggregate timeseries
        """
        # default subregions to all regions other than `region`
        if subregions is None:
            rows = self._apply_filters(variable=variable)
            subregions = set(self.data[rows].region) - set([region])

        if not len(subregions):
            msg = 'cannot aggregate variable `{}` to `{}` because it does not'\
                  ' exist in any subregion'
            logger().info(msg.format(variable, region))

            return

        # compute aggregate over all subregions
        subregion_df = self.filter(region=subregions)
        cols = ['region', 'variable']
        _data = _aggregate(subregion_df.filter(variable=variable).data, cols)

        # add components at the `region` level, defaults to all variables one
        # level below `variable` that are only present in `region`
        region_df = self.filter(region=region)
        components = components or (
            set(region_df._variable_components(variable)).difference(
                subregion_df._variable_components(variable)))

        if len(components):
            rows = region_df._apply_filters(variable=components)
            _data = _data.add(_aggregate(region_df.data[rows], cols),
                              fill_value=0)

        if append is True:
            self.append(_data, region=region, variable=variable, inplace=True)
        else:
            return _data