def igphyml(input_file=None, tree_file=None, root=None, verbose=False):
    '''
    Computes a phylogenetic tree using IgPhyML.

    .. note::
        
        IgPhyML must be installed. It can be downloaded from https://github.com/kbhoehn/IgPhyML.

    Args:

        input_file (str): Path to a Phylip-formatted multiple sequence alignment. Required.

        tree_file (str): Path to the output tree file.

        root (str): Name of the root sequence. Required.

        verbose (bool): If `True`, prints the standard output and standard error for each IgPhyML run. 
            Default is `False`.
    '''

    if shutil.which('igphyml') is None:
        raise RuntimeError('It appears that IgPhyML is not installed.\nPlease install and try again.')
    
    # first, tree topology is estimated with the M0/GY94 model
    igphyml_cmd1 = 'igphyml -i {} -m GY -w M0 -t e --run_id gy94'.format(aln_file)
    p1 = sp.Popen(igphyml_cmd1, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout1, stderr1 = p1.communicate()
    if verbose:
        print(stdout1 + '\n')
        print(stderr1 + '\n\n')
    intermediate = input_file + '_igphyml_tree.txt_gy94'

    # now  we fit the HLP17 model once the tree topology is fixed
    igphyml_cmd2 = 'igphyml -i {0} -m HLP17 --root {1} -o lr -u {}_igphyml_tree.txt_gy94 -o {}'.format(input_file,
                                                                                                       root,
                                                                                                       tree_file)
    p2 = sp.Popen(igphyml_cmd2, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout2, stderr2 = p2.communicate()
    if verbose:
        print(stdout2 + '\n')
        print(stderr2 + '\n')
    return tree_file + '_igphyml_tree.txt'