def chempot_vs_gamma(self, ref_delu, chempot_range, miller_index=(),
                         delu_dict={}, delu_default=0, JPERM2=False,
                         show_unstable=False, ylim=[], plt=None,
                         no_clean=False, no_doped=False,
                         use_entry_labels=False, no_label=False):
        """
        Plots the surface energy as a function of chemical potential.
            Each facet will be associated with its own distinct colors.
            Dashed lines will represent stoichiometries different from that
            of the mpid's compound. Transparent lines indicates adsorption.

        Args:
            ref_delu (sympy Symbol): The range stability of each slab is based
                on the chempot range of this chempot. Should be a sympy Symbol
                object of the format: Symbol("delu_el") where el is the name of
                the element
            chempot_range ([max_chempot, min_chempot]): Range to consider the
                stability of the slabs.
            miller_index (list): Miller index for a specific facet to get a
                dictionary for.
            delu_dict (Dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            JPERM2 (bool): Whether to plot surface energy in /m^2 (True) or
                eV/A^2 (False)
            show_unstable (bool): Whether or not to show parts of the surface
                energy plot outside the region of stability.
            ylim ([ymax, ymin]): Range of y axis
            no_doped (bool): Whether to plot for the clean slabs only.
            no_clean (bool): Whether to plot for the doped slabs only.
            use_entry_labels (bool): If True, will label each slab configuration
                according to their given label in the SlabEntry object.
            no_label (bool): Option to turn off labels.

        Returns:
            (Plot): Plot of surface energy vs chempot for all entries.
        """

        chempot_range = sorted(chempot_range)

        plt = pretty_plot(width=8, height=7) if not plt else plt
        axes = plt.gca()

        for hkl in self.all_slab_entries.keys():
            if miller_index and hkl != tuple(miller_index):
                continue
            # Get the chempot range of each surface if we only
            # want to show the region where each slab is stable
            if not show_unstable:
                stable_u_range_dict = self.stable_u_range_dict(chempot_range, ref_delu,
                                                               no_doped=no_doped,
                                                               delu_dict=delu_dict,
                                                               miller_index=hkl)

            already_labelled = []
            label = ''
            for clean_entry in self.all_slab_entries[hkl]:

                urange = stable_u_range_dict[clean_entry] if \
                    not show_unstable else chempot_range
                # Don't plot if the slab is unstable, plot if it is.
                if urange != []:

                    label = clean_entry.label
                    if label in already_labelled:
                        label = None
                    else:
                        already_labelled.append(label)
                    if not no_clean:
                        if use_entry_labels:
                            label = clean_entry.label
                        if no_label:
                            label = ""
                        plt = self.chempot_vs_gamma_plot_one(plt, clean_entry, ref_delu,
                                                             urange, delu_dict=delu_dict,
                                                             delu_default=delu_default,
                                                             label=label, JPERM2=JPERM2)
                if not no_doped:
                    for ads_entry in self.all_slab_entries[hkl][clean_entry]:
                        # Plot the adsorbed slabs
                        # Generate a label for the type of slab
                        urange = stable_u_range_dict[ads_entry] \
                            if not show_unstable else chempot_range
                        if urange != []:
                            if use_entry_labels:
                                label = ads_entry.label
                            if no_label:
                                label = ""
                            plt = self.chempot_vs_gamma_plot_one(plt, ads_entry,
                                                                 ref_delu, urange,
                                                                 delu_dict=delu_dict,
                                                                 delu_default=delu_default,
                                                                 label=label,
                                                                 JPERM2=JPERM2)

        # Make the figure look nice
        plt.ylabel(r"Surface energy (J/$m^{2}$)") if JPERM2 \
            else plt.ylabel(r"Surface energy (eV/$\AA^{2}$)")
        plt = self.chempot_plot_addons(plt, chempot_range, str(ref_delu).split("_")[1],
                                       axes, ylim=ylim)

        return plt