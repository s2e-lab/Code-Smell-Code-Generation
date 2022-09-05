def run_count(bam_file, dexseq_gff, stranded, out_file, data):
    """
    run dexseq_count on a BAM file
    """
    assert file_exists(bam_file), "%s does not exist." % bam_file
    sort_order = bam._get_sort_order(bam_file, {})
    assert sort_order, "Cannot determine sort order of %s." % bam_file
    strand_flag = _strand_flag(stranded)
    assert strand_flag, "%s is not a valid strandedness value." % stranded
    if not dexseq_gff:
        logger.info("No DEXSeq GFF file was found, skipping exon-level counting.")
        return None
    elif not file_exists(dexseq_gff):
        logger.info("%s was not found, so exon-level counting is being "
                    "skipped." % dexseq_gff)
        return None

    dexseq_count = _dexseq_count_path()
    if not dexseq_count:
        logger.info("DEXseq is not installed, skipping exon-level counting.")
        return None

    if dd.get_aligner(data) == "bwa":
        logger.info("Can't use DEXSeq with bwa alignments, skipping exon-level counting.")
        return None

    sort_flag = "name" if sort_order == "queryname" else "pos"
    is_paired = bam.is_paired(bam_file)
    paired_flag = "yes" if is_paired else "no"
    bcbio_python = sys.executable

    if file_exists(out_file):
        return out_file
    cmd = ("{bcbio_python} {dexseq_count} -f bam -r {sort_flag} -p {paired_flag} "
           "-s {strand_flag} {dexseq_gff} {bam_file} {tx_out_file}")
    message = "Counting exon-level counts with %s and %s." % (bam_file, dexseq_gff)
    with file_transaction(data, out_file) as tx_out_file:
        do.run(cmd.format(**locals()), message)
    return out_file