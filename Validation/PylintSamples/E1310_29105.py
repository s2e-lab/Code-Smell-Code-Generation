def view_structure(self, only_chains=None, opacity=1.0, recolor=False, gui=False):
        """Use NGLviewer to display a structure in a Jupyter notebook

        Args:
            only_chains (str, list): Chain ID or IDs to display
            opacity (float): Opacity of the structure
            recolor (bool): If structure should be cleaned and recolored to silver
            gui (bool): If the NGLview GUI should show up

        Returns:
            NGLviewer object

        """
        # TODO: show_structure_file does not work for MMTF files - need to check for that and load accordingly

        if ssbio.utils.is_ipynb():
            import nglview as nv
        else:
            raise EnvironmentError('Unable to display structure - not running in a Jupyter notebook environment')

        if not self.structure_file:
            raise ValueError("Structure file not loaded")

        only_chains = ssbio.utils.force_list(only_chains)
        to_show_chains = '( '
        for c in only_chains:
            to_show_chains += ':{} or'.format(c)
        to_show_chains = to_show_chains.strip(' or ')
        to_show_chains += ' )'

        if self.file_type == 'mmtf' or self.file_type == 'mmtf.gz':
            view = nv.NGLWidget()
            view.add_component(self.structure_path)
        else:
            view = nv.show_structure_file(self.structure_path, gui=gui)

        if recolor:
            view.clear_representations()
            if only_chains:
                view.add_cartoon(selection='{} and (not hydrogen)'.format(to_show_chains), color='silver', opacity=opacity)
            else:
                view.add_cartoon(selection='protein', color='silver', opacity=opacity)
        elif only_chains:
            view.clear_representations()
            view.add_cartoon(selection='{} and (not hydrogen)'.format(to_show_chains), color='silver', opacity=opacity)

        return view