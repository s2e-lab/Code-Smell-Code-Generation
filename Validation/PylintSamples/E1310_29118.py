def add_residues_highlight_to_nglview(view, structure_resnums, chain, res_color='red'):
    """Add a residue number or numbers to an NGLWidget view object.

    Args:
        view (NGLWidget): NGLWidget view object
        structure_resnums (int, list): Residue number(s) to highlight, structure numbering
        chain (str, list): Chain ID or IDs of which residues are a part of. If not provided, all chains in the
            mapped_chains attribute will be used. If that is also empty, and exception is raised.
        res_color (str): Color to highlight residues with

    """
    chain = ssbio.utils.force_list(chain)

    if isinstance(structure_resnums, list):
        structure_resnums = list(set(structure_resnums))
    elif isinstance(structure_resnums, int):
        structure_resnums = ssbio.utils.force_list(structure_resnums)
    else:
        raise ValueError('Input must either be a residue number of a list of residue numbers')

    to_show_chains = '( '
    for c in chain:
        to_show_chains += ':{} or'.format(c)
    to_show_chains = to_show_chains.strip(' or ')
    to_show_chains += ' )'

    to_show_res = '( '
    for m in structure_resnums:
        to_show_res += '{} or '.format(m)
    to_show_res = to_show_res.strip(' or ')
    to_show_res += ' )'

    log.info('Selection: {} and not hydrogen and {}'.format(to_show_chains, to_show_res))

    view.add_ball_and_stick(selection='{} and not hydrogen and {}'.format(to_show_chains, to_show_res), color=res_color)