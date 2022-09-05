def get_protein_dict(cif_data):
    """ Parse cif_data dict for a subset of its data.

    Notes
    -----
    cif_data dict contains all the data from the .cif file, with values as strings.
    This function returns a more 'human readable' dictionary of key-value pairs.
    The keys have simpler (and still often more descriptive!) names, and the values are not restricted to being strings.
    To add more key-value pairs to the protein_dict, follow the patterns used in this function.
    Add the key and youre name for it to mmcif_data_names.
    Will it need further parsing, like with the dates in the function below?
    If the value is not a string, add it to a list of data-types at the end of the function.
    More information on what key-value pairs can be obtained can be gleaned by examining cif_data and/or by viewing the
    mmcif resource on the PDB website: http://mmcif.wwpdb.org/docs/pdb_to_pdbx_correspondences.html
    WARNING: Do not alter the keys of protein_dict without caution.
    The keys of protein_dict MUST match the column names of the Protein model in the protgraph database.

    Parameters
    ----------
    cif_data : dict
        Key/value pairs taken directly from a .cif file.
        Output of the function dict_from_mmcif.

    Returns
    -------
    protein_dict : dict
        A dictionary containing a parsed subset of the data in cif_data.
        The keys have the same name as fields in the Protein model.
    """

    # Dictionary relating the keys of protein_dict (column names in Protein model) to the keys of cif_data.
    mmcif_data_names = {
        'keywords': '_struct_keywords.text',
        'header': '_struct_keywords.pdbx_keywords',
        'space_group': '_symmetry.space_group_name_H-M',
        'experimental_method': '_exptl.method',
        'crystal_growth': '_exptl_crystal_grow.pdbx_details',
        'resolution': '_refine.ls_d_res_high',
        'r_value_obs': '_refine.ls_R_factor_obs',
        'atoms_protein': '_refine_hist.pdbx_number_atoms_protein',
        'atoms_solvent': '_refine_hist.number_atoms_solvent',
        'atoms_ligand': '_refine_hist.pdbx_number_atoms_ligand',
        'atoms_nucleic_acid': '_refine_hist.pdbx_number_atoms_nucleic_acid',
        'atoms_total': '_refine_hist.number_atoms_total',
        'title': '_struct.title',
        'pdb_descriptor': '_struct.pdbx_descriptor',
        'model_details': '_struct.pdbx_model_details',
        'casp_flag': '_struct.pdbx_CASP_flag',
        'model_type_details': '_struct.pdbx_model_type_details',
        'ncbi_taxonomy': '_entity_src_nat.pdbx_ncbi_taxonomy_id',
        'ncbi_taxonomy_gene': '_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id',
        'ncbi_taxonomy_host_org': '_entity_src_gen.pdbx_host_org_ncbi_taxonomy_id',
    }

    # Set up initial protein_dict.
    protein_dict = {}
    for column_name, cif_name in mmcif_data_names.items():
        try:
            data = cif_data[cif_name]
        except IndexError:
            data = None
        except KeyError:
            data = None
        protein_dict[column_name] = data

    # These entries are modified from the mmcif dictionary.
    # There may be many revision dates in cif_data. We save the original deposition, release and last_modified dates.
    # If there are many dates, they will be in a list in cif_data, otherwise it's one date in a string
    # Is there a tidier way to do this?
    if isinstance(cif_data['_database_PDB_rev.date_original'], str):
        protein_dict['deposition_date'] = cif_data['_database_PDB_rev.date_original']
    else:
        protein_dict['deposition_date'] = cif_data['_database_PDB_rev.date_original'][0]
    if isinstance(cif_data['_database_PDB_rev.date'], str):
        protein_dict['release_date'] = cif_data['_database_PDB_rev.date']
        protein_dict['last_modified_date'] = cif_data['_database_PDB_rev.date']
    else:
        protein_dict['release_date'] = cif_data['_database_PDB_rev.date'][0]
        protein_dict['last_modified_date'] = cif_data['_database_PDB_rev.date'][-1]

    # crystal_growth should be a string or None
    crystal_growth = protein_dict['crystal_growth']
    if type(crystal_growth) == list and len(crystal_growth) >= 1:
        protein_dict['crystal_growth'] = crystal_growth[0]
    else:
        protein_dict['crystal_growth'] = None

    # taxonomy data types should be ints, not lists
    taxonomy_keys = ['ncbi_taxonomy', 'ncbi_taxonomy_gene', 'ncbi_taxonomy_host_org']
    for taxonomy_key in taxonomy_keys:
        if protein_dict[taxonomy_key]:
            if type(protein_dict[taxonomy_key]) == list:
                try:
                    protein_dict[taxonomy_key] = int(protein_dict[taxonomy_key][0])
                except ValueError or IndexError:
                    protein_dict[taxonomy_key] = None

    # Convert data types from strings to their correct data type.
    ints = ['atoms_ligand', 'atoms_nucleic_acid', 'atoms_protein', 'atoms_solvent', 'atoms_total']
    floats = ['r_value_obs', 'resolution']
    dates = ['deposition_date', 'release_date', 'last_modified_date']

    for k, v in protein_dict.items():
        if v:
            if v == '?' or v == 'None' or v == '.':
                protein_dict[k] = None
            elif k in ints:
                protein_dict[k] = int(v)
            elif k in floats:
                protein_dict[k] = float(v)
            elif k in dates:
                protein_dict[k] = datetime.datetime.strptime(v, '%Y-%m-%d')
            # Parse awkward strings from cif_data.
            elif type(v) == str:
                v = v.replace('loop_', '')
                v = v.replace(' # ', '')
                if v[0] == v[-1] == '\'':
                    protein_dict[k] = v[1:-1]

    return protein_dict