def get_rna_counts(rna_file, transcript_name):
    """Get coverage for a given RNA BAM file, return read counts. """
    # check if the RNA file exists
    if not os.path.exists(rna_file):
        msg = 'RNA-Seq BAM file "{}" does not exist'.format(rna_file)
        logging.error(msg)
        raise OSError(msg)
    rna_counts = {}

    cov_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        subprocess.check_call(
            ['bedtools', 'genomecov', '-ibam', rna_file,
             '-bg'], stdout=cov_file)
    except subprocess.CalledProcessError as e:
        # needs testing
        raise ribocore.RNACountsError('Could not generate coverage for RNA BAM file. \n{}\n'.format(e))
    for line in open(cov_file.name):
        line = line.split()
        if line[0] == transcript_name:
            position, count = int(line[1]) + 1, int(line[3])
            rna_counts[position] = count
    cov_file.close()
    os.unlink(cov_file.name)
    return rna_counts