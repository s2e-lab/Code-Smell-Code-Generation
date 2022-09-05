def sort_bam_by_reference(job, job_vars):
    """
    Sorts the bam by reference

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    # Unpack variables
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    # I/O
    sorted_bam, sorted_bai = return_input_paths(job, work_dir, ids, 'sorted.bam', 'sorted.bam.bai')
    output = os.path.join(work_dir, 'sort_by_ref.bam')
    # Call: Samtools
    ref_seqs = []
    handle = subprocess.Popen(["samtools", "view", "-H", sorted_bam], stdout=subprocess.PIPE).stdout
    for line in handle:
        if line.startswith("@SQ"):
            tmp = line.split("\t")
            chrom = tmp[1].split(":")[1]
            ref_seqs.append(chrom)
    handle.close()
    # Iterate through chromosomes to create mini-bams
    for chrom in ref_seqs:
        # job.addChildJobFn(sbbr_child, chrom, os.path.join(work_dir, chrom), sorted_bam)
        cmd_view = ["samtools", "view", "-b", sorted_bam, chrom]
        cmd_sort = ["samtools", "sort", "-m", "3000000000", "-n", "-", os.path.join(work_dir, chrom)]
        p1 = subprocess.Popen(cmd_view, stdout=subprocess.PIPE)
        subprocess.check_call(cmd_sort, stdin=p1.stdout)
    sorted_files = [os.path.join(work_dir, chrom) + '.bam' for chrom in ref_seqs]
    cmd = ["samtools", "cat", "-o", output] + sorted_files
    subprocess.check_call(cmd)
    # Write to FileStore
    ids['sort_by_ref.bam'] = job.fileStore.writeGlobalFile(output)
    rsem_id = job.addChildJobFn(transcriptome, job_vars, disk='30 G', memory='30 G').rv()
    exon_id = job.addChildJobFn(exon_count, job_vars, disk='30 G').rv()
    return exon_id, rsem_id