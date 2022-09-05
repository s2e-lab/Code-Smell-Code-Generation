def read_variants(fns, remove=['DBSNP'], keep_only=True,
                  min_tumor_f=0.1, min_tumor_cov=14,
                  min_normal_cov=8):
    """Read muTect results from the list of files fns

    Parameters
    ----------
    fns : list
        List of MuTect output files.

    Returns
    -------
    variants : pandas.DataFrame
        Pandas DataFrame summarizing variant calling results.

    remove : list
        List of site types for column "dbsnp_site" to remove.

    keep_only : boolean
        If True, only keep variants with 'KEEP' in "judgement" column.
        Otherwise, keep all variants.

    min_tumor_f : float between 0 and 1
        Minimum tumor allelic fraction.

    min_tumor_cov : int > 0
        Minimum coverage of the variant in the tumor.

    min_normal_cov : int > 0
        Minimum coverage of the variant in the normal.

    """
    variants = []
    for i, f in enumerate(fns):
        # If keep_only, use awk to only grab those lines for big speedup.
        if keep_only:
            from numpy import dtype
            import subprocess
            res = subprocess.check_output(
                'awk \'$35 == "KEEP"\' {}'.format(f), shell=True)
            if res.strip() != '': 
                columns = [u'contig', u'position', u'context', u'ref_allele',
                           u'alt_allele', u'tumor_name', u'normal_name',
                           u'score', u'dbsnp_site', u'covered', u'power',
                           u'tumor_power', u'normal_power', u'total_pairs',
                           u'improper_pairs', u'map_Q0_reads', u't_lod_fstar',
                           u'tumor_f', u'contaminant_fraction',
                           u'contaminant_lod', u't_ref_count', u't_alt_count',
                           u't_ref_sum', u't_alt_sum', u't_ref_max_mapq',
                           u't_alt_max_mapq', u't_ins_count', u't_del_count',
                           u'normal_best_gt', u'init_n_lod', u'n_ref_count',
                           u'n_alt_count', u'n_ref_sum', u'n_alt_sum',
                           u'judgement']
                tdf = pd.DataFrame(
                    [x.split('\t') for x in res.strip().split('\n')],
                    columns=columns)
                tdf = tdf.convert_objects(convert_numeric=True)
            else:
                tdf = pd.DataFrame(columns=columns)
            tdf['contig'] = tdf.contig.astype(object)

        else:
            tdf = pd.read_table(f, index_col=None, header=0, skiprows=1,
                                low_memory=False, 
                                dtype={'contig':object})
        for t in remove:
            tdf = tdf[tdf.dbsnp_site != t]
        tdf = tdf[tdf.tumor_f > min_tumor_f]
        tdf = tdf[tdf.t_ref_count + tdf.t_alt_count > min_tumor_cov]
        tdf = tdf[tdf.n_ref_count + tdf.n_alt_count > min_normal_cov]
        variants.append(tdf)
    variants = pd.concat(variants)
    variants.index = range(variants.shape[0])
    return variants