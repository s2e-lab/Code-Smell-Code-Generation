def getTmpFilename(self, tmp_dir="/tmp",prefix='tmp',suffix='.fasta',\
           include_class_id=False,result_constructor=FilePath):
        """ Define Tmp filename to contain .fasta suffix, since pplacer requires
            the suffix to be .fasta """

        return super(Pplacer,self).getTmpFilename(tmp_dir=tmp_dir,
                                    prefix=prefix,
                                    suffix=suffix,
                                    include_class_id=include_class_id,
                                    result_constructor=result_constructor)