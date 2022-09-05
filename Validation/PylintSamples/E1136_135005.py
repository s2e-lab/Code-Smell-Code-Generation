def plot_zt_dop(self, temps='all', output='average', relaxation_time=1e-14):
        """
        Plot the figure of merit zT in function of doping levels for different
        temperatures.

        Args:
            temps: the default 'all' plots all the temperatures in the analyzer.
                   Specify a list of temperatures if you want to plot only some.
            output: with 'average' you get an average of the three directions
                    with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value
        
        Returns:
            a matplotlib object
        """
        import matplotlib.pyplot as plt
        if output == 'average':
            zt = self._bz.get_zt(relaxation_time=relaxation_time,
                                 output='average')
        elif output == 'eigs':
            zt = self._bz.get_zt(relaxation_time=relaxation_time, output='eigs')

        tlist = sorted(zt['n'].keys()) if temps == 'all' else temps
        plt.figure(figsize=(22, 14))
        for i, dt in enumerate(['n', 'p']):
            plt.subplot(121 + i)
            for temp in tlist:
                if output == 'eigs':
                    for xyz in range(3):
                        plt.semilogx(self._bz.doping[dt],
                                     zip(*zt[dt][temp])[xyz],
                                     marker='s',
                                     label=str(xyz) + ' ' + str(temp) + ' K')
                elif output == 'average':
                    plt.semilogx(self._bz.doping[dt], zt[dt][temp],
                                 marker='s', label=str(temp) + ' K')
            plt.title(dt + '-type', fontsize=20)
            if i == 0:
                plt.ylabel("zT", fontsize=30.0)
            plt.xlabel('Doping concentration ($cm^{-3}$)', fontsize=30.0)

            p = 'lower right' if i == 0 else ''
            plt.legend(loc=p, fontsize=15)
            plt.grid()
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)

        plt.tight_layout()

        return plt