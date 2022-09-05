def register_default_dimensions(cube, slvr_cfg):
    """ Register the default dimensions for a RIME solver """

    import montblanc.src_types as mbs

    # Pull out the configuration options for the basics
    autocor = slvr_cfg['auto_correlations']

    ntime = 10
    na = 7
    nbands = 1
    nchan = 16
    npol = 4

    # Infer number of baselines from number of antenna,
    nbl = nr_of_baselines(na, autocor)

    if not npol == 4:
        raise ValueError("npol set to {}, but only 4 polarisations "
                         "are currently supported.")

    # Register these dimensions on this solver.
    cube.register_dimension('ntime', ntime,
        description="Timesteps")
    cube.register_dimension('na', na,
        description="Antenna")
    cube.register_dimension('nbands', nbands,
        description="Bands")
    cube.register_dimension('nchan', nchan,
        description="Channels")
    cube.register_dimension('npol', npol,
        description="Polarisations")
    cube.register_dimension('nbl', nbl,
        description="Baselines")

    # Register dependent dimensions
    cube.register_dimension('npolchan', nchan*npol,
        description='Polarised channels')
    cube.register_dimension('nvis', ntime*nbl*nchan,
        description='Visibilities')

    # Convert the source types, and their numbers
    # to their number variables and numbers
    # { 'point':10 } => { 'npsrc':10 }
    src_cfg = default_sources()
    src_nr_vars = sources_to_nr_vars(src_cfg)
    # Sum to get the total number of sources
    cube.register_dimension('nsrc', sum(src_nr_vars.itervalues()),
        description="Sources (Total)")

    # Register the individual source types
    for nr_var, nr_of_src in src_nr_vars.iteritems():
        cube.register_dimension(nr_var, nr_of_src,
            description='{} sources'.format(mbs.SOURCE_DIM_TYPES[nr_var]))