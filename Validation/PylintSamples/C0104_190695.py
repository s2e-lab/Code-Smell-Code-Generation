def destroy(force):
    """Destroy all indexes."""
    click.secho('Destroying indexes...', fg='red', bold=True, file=sys.stderr)
    with click.progressbar(
            current_search.delete(ignore=[400, 404] if force else None),
            length=current_search.number_of_indexes) as bar:
        for name, response in bar:
            bar.label = name