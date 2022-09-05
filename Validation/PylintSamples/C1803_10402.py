def from_bam(pysam_samfile, loci, normalized_contig_names=True):
        '''
        Create a PileupCollection for a set of loci from a BAM file.

        Parameters
        ----------
        pysam_samfile : `pysam.Samfile` instance, or filename string
            to a BAM file. The BAM file must be indexed.

        loci : list of Locus instances
            Loci to collect pileups for.

        normalized_contig_names : whether the contig names have been normalized
            (e.g. pyensembl removes the 'chr' prefix). Set to true to
            de-normalize the names when querying the BAM file.

        Returns
        ----------
        PileupCollection instance containing pileups for the specified loci.
        All alignments in the BAM file are included (e.g. duplicate reads,
        secondary alignments, etc.). See `PileupCollection.filter` if these
        need to be removed. 
        '''

        loci = [to_locus(obj) for obj in loci]

        close_on_completion = False
        if typechecks.is_string(pysam_samfile):
            pysam_samfile = Samfile(pysam_samfile)
            close_on_completion = True        

        try:
            # Map from pyensembl normalized chromosome names used in Variant to
            # the names used in the BAM file.
            if normalized_contig_names:
                chromosome_name_map = {}
                for name in pysam_samfile.references:
                    normalized = pyensembl.locus.normalize_chromosome(name)
                    chromosome_name_map[normalized] = name
                    chromosome_name_map[name] = name
            else:
                chromosome_name_map = None

            result = PileupCollection({})

            # Optimization: we sort variants so our BAM reads are localized.
            locus_iterator = itertools.chain.from_iterable(
                (Locus.from_interbase_coordinates(locus_interval.contig, pos)
                    for pos
                    in locus_interval.positions)
                for locus_interval in sorted(loci))
            for locus in locus_iterator:
                result.pileups[locus] = Pileup(locus, [])
                if normalized_contig_names:
                    try:
                        chromosome = chromosome_name_map[locus.contig]
                    except KeyError:
                        logging.warn("No such contig in bam: %s" % locus.contig)
                        continue
                else:
                    chromosome = locus.contig
                columns = pysam_samfile.pileup(
                    chromosome,
                    locus.position,
                    locus.position + 1,  # exclusive, 0-indexed
                    truncate=True,
                    stepper="nofilter")
                try:
                    column = next(columns)
                except StopIteration:
                    # No reads align to this locus.
                    continue

                # Note that storing the pileups here is necessary, since the
                # subsequent assertion will invalidate our column.
                pileups = column.pileups
                assert list(columns) == []  # column is invalid after this.
                for pileup_read in pileups:
                    if not pileup_read.is_refskip:
                        element = PileupElement.from_pysam_alignment(
                            locus, pileup_read)
                        result.pileups[locus].append(element)
            return result
        finally:
            if close_on_completion:
                pysam_samfile.close()