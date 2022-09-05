def _ensure_paths_and_types(index: Dict[str, str]) -> Dict[str, Path]:
    """ Take the direct results of loading the config and make sure
    the filesystem reflects them.
    """
    configs_by_name = {ce.name: ce for ce in CONFIG_ELEMENTS}
    correct_types: Dict[str, Path] = {}
    for key, item in index.items():
        if key not in configs_by_name:  # old config, ignore
            continue
        if configs_by_name[key].kind == ConfigElementType.FILE:
            it = Path(item)
            it.parent.mkdir(parents=True, exist_ok=True)
            correct_types[key] = it
        elif configs_by_name[key].kind == ConfigElementType.DIR:
            it = Path(item)
            it.mkdir(parents=True, exist_ok=True)
            correct_types[key] = it
        else:
            raise RuntimeError(
                f"unhandled kind in ConfigElements: {key}: "
                f"{configs_by_name[key].kind}")
    return correct_types