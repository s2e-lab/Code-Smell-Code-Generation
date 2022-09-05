def sort_sam(sam, sort):
    """
    sort sam file
    """
    tempdir = '%s/' % (os.path.abspath(sam).rsplit('/', 1)[0])
    if sort is True:
        mapping = '%s.sorted.sam' % (sam.rsplit('.', 1)[0])
        if sam != '-':
            if os.path.exists(mapping) is False:
                os.system("\
                    sort -k1 --buffer-size=%sG -T %s -o %s %s\
                    " % (sbuffer, tempdir, mapping, sam)) 
        else:
            mapping = 'stdin-sam.sorted.sam'
            p = Popen("sort -k1 --buffer-size=%sG -T %s -o %s" \
                    % (sbuffer, tempdir, mapping), stdin = sys.stdin, shell = True) 
            p.communicate()
        mapping = open(mapping)
    else:
        if sam == '-':
            mapping = sys.stdin
        else:
            mapping = open(sam)
    return mapping