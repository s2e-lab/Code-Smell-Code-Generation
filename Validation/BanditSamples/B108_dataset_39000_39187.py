def config(remote_base=       'https://raw.githubusercontent.com/SciCrunch/NIF-Ontology/',
           local_base=        None,  # devconfig.ontology_local_repo by default
           branch=            devconfig.neurons_branch,
           core_graph_paths= ['ttl/phenotype-core.ttl',
                              'ttl/phenotypes.ttl'],
           core_graph=        None,
           in_graph_paths=    tuple(),
           out_graph_path=    '/tmp/_Neurons.ttl',
           out_imports=      ['ttl/phenotype-core.ttl'],
           out_graph=         None,
           prefixes=          tuple(),
           force_remote=      False,
           checkout_ok=       ont_checkout_ok,
           scigraph=          None,  # defaults to devconfig.scigraph_api
           iri=               None,
           sources=           tuple(),
           source_file=       None,
           use_local_import_paths=True,
           ignore_existing=   True):
    """ Wraps graphBase.configGraphIO to provide a set of sane defaults
        for input ontologies and output files. """
    graphBase.configGraphIO(remote_base=remote_base,
                            local_base=local_base,
                            branch=branch,
                            core_graph_paths=core_graph_paths,
                            core_graph=core_graph,
                            in_graph_paths=in_graph_paths,
                            out_graph_path=out_graph_path,
                            out_imports=out_imports,
                            out_graph=out_graph,
                            prefixes=prefixes,
                            force_remote=force_remote,
                            checkout_ok=checkout_ok,
                            scigraph=scigraph,
                            iri=iri,
                            sources=sources,
                            source_file=source_file,
                            use_local_import_paths=use_local_import_paths,
                            ignore_existing=ignore_existing)

    pred = graphBase._predicates
    return pred