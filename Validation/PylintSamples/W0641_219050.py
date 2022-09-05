def run(_, data, out_dir=None):
    """Run samtools stats with reports on mapped reads, duplicates and insert sizes.
    """
    stats_file, idxstats_file = _get_stats_files(data, out_dir)
    samtools = config_utils.get_program("samtools", data["config"])
    bam_file = dd.get_align_bam(data) or dd.get_work_bam(data)
    if not utils.file_exists(stats_file):
        utils.safe_makedir(out_dir)
        with file_transaction(data, stats_file) as tx_out_file:
            cores = dd.get_num_cores(data)
            cmd = "{samtools} stats -@ {cores} {bam_file}"
            cmd += " > {tx_out_file}"
            do.run(cmd.format(**locals()), "samtools stats", data)
    if not utils.file_exists(idxstats_file):
        utils.safe_makedir(out_dir)
        with file_transaction(data, idxstats_file) as tx_out_file:
            cmd = "{samtools} idxstats {bam_file}"
            cmd += " > {tx_out_file}"
            do.run(cmd.format(**locals()), "samtools index stats", data)
    out = {"base": idxstats_file, "secondary": [stats_file]}
    out["metrics"] = _parse_samtools_stats(stats_file)
    return out