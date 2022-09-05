def rmarkdown_draft(filename, template, package):
    """
    create a draft rmarkdown file from an installed template
    """
    if file_exists(filename):
        return filename
    draft_template = Template(
        'rmarkdown::draft("$filename", template="$template", package="$package", edit=FALSE)'
    )
    draft_string = draft_template.substitute(
        filename=filename, template=template, package=package)
    report_dir = os.path.dirname(filename)
    rcmd = Rscript_cmd()
    with chdir(report_dir):
        do.run([rcmd, "--no-environ", "-e", draft_string], "Creating bcbioRNASeq quality control template.")
        do.run(["sed", "-i", "s/YYYY-MM-DD\///g", filename], "Editing bcbioRNAseq quality control template.")
    return filename