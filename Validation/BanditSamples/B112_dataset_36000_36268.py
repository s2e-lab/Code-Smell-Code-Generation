def make_clusters(span_tree, cut_value):
    """ Find clusters from the spanning tree

    Parameters
    ----------
    span_tree : a sparse nsrcs x nsrcs array
       Filled with zeros except for the active edges, which are filled with the
       edge measures (either distances or sigmas

    cut_value : float
       Value used to cluster group.  All links with measures above this calue will be cut.

    returns dict(int:[int,...])  
       A dictionary of clusters.   Each cluster is a source index and the list of other sources in the cluster.    
    """
    iv0, iv1 = span_tree.nonzero()

    # This is the dictionary of all the pairings for each source
    match_dict = {}

    for i0, i1 in zip(iv0, iv1):
        d = span_tree[i0, i1]
        # Cut on the link distance
        if d > cut_value:
            continue

        imin = int(min(i0, i1))
        imax = int(max(i0, i1))
        if imin in match_dict:
            match_dict[imin][imax] = True
        else:
            match_dict[imin] = {imax: True}

    working = True
    while working:

        working = False
        rev_dict = make_rev_dict_unique(match_dict)
        k_sort = rev_dict.keys()
        k_sort.sort()
        for k in k_sort:
            v = rev_dict[k]
            # Multiple mappings
            if len(v) > 1:
                working = True
                v_sort = v.keys()
                v_sort.sort()
                cluster_idx = v_sort[0]
                for vv in v_sort[1:]:
                    try:
                        to_merge = match_dict.pop(vv)
                    except:
                        continue
                    try:
                        match_dict[cluster_idx].update(to_merge)
                        match_dict[cluster_idx][vv] = True
                    except:
                        continue
                    # remove self references
                    try:
                        match_dict[cluster_idx].pop(cluster_idx)
                    except:
                        pass

    # Convert to a int:list dictionary
    cdict = {}
    for k, v in match_dict.items():
        cdict[k] = v.keys()

    # make the reverse dictionary
    rdict = make_reverse_dict(cdict)
    return cdict, rdict