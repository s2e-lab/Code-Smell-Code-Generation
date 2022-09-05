def load(name, path=None, strip=None, verb=True):
    """     Load a tofu object file

    Can load from .npz or .txt files
        In future versions, will also load from .mat

    The file must have been saved with tofu (i.e.: must be tofu-formatted)
    The associated tofu object will be created and returned

    Parameters
    ----------
    name:   str
        Name of the file to load from, can include the path
    path:   None / str
        Path where the file is located (if not provided in name), defaults './'
    strip:  None / int
        FLag indicating whether to strip the object of some attributes
            => see the docstring of the class strip() method for details
    verb:   bool
        Flag indocating whether to print a summary of the loaded file
    """

    lmodes = ['.npz','.mat','.txt']
    name, mode, pfe = _filefind(name=name, path=path, lmodes=lmodes)

    if mode == 'txt':
        obj = _load_from_txt(name, pfe)
    else:
        if mode == 'npz':
            dd = _load_npz(pfe)
        elif mode == 'mat':
            dd = _load_mat(pfe)

        # Recreate from dict
        exec("import tofu.{0} as mod".format(dd['dId_dall_Mod']))
        obj = eval("mod.{0}(fromdict=dd)".format(dd['dId_dall_Cls']))

    if strip is not None:
        obj.strip(strip=strip)

    # print
    if verb:
        msg = "Loaded from:\n"
        msg += "    "+pfe
        print(msg)
    return obj