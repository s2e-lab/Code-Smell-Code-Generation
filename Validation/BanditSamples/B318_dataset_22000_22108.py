def DocbookMan(env, target, source=None, *args, **kw):
    """
    A pseudo-Builder, providing a Docbook toolchain for Man page output.
    """
    # Init list of targets/sources
    target, source = __extend_targets_sources(target, source)

    # Init XSL stylesheet
    __init_xsl_stylesheet(kw, env, '$DOCBOOK_DEFAULT_XSL_MAN', ['manpages','docbook.xsl'])

    # Setup builder
    __builder = __select_builder(__lxml_builder, __libxml2_builder, __xsltproc_builder)

    # Create targets
    result = []
    for t,s in zip(target,source):
        volnum = "1"
        outfiles = []
        srcfile = __ensure_suffix(str(s),'.xml')
        if os.path.isfile(srcfile):
            try:
                import xml.dom.minidom
                
                dom = xml.dom.minidom.parse(__ensure_suffix(str(s),'.xml'))
                # Extract volume number, default is 1
                for node in dom.getElementsByTagName('refmeta'):
                    for vol in node.getElementsByTagName('manvolnum'):
                        volnum = __get_xml_text(vol)
                        
                # Extract output filenames
                for node in dom.getElementsByTagName('refnamediv'):
                    for ref in node.getElementsByTagName('refname'):
                        outfiles.append(__get_xml_text(ref)+'.'+volnum)
                        
            except:
                # Use simple regex parsing 
                f = open(__ensure_suffix(str(s),'.xml'), 'r')
                content = f.read()
                f.close()
                
                for m in re_manvolnum.finditer(content):
                    volnum = m.group(1)
                    
                for m in re_refname.finditer(content):
                    outfiles.append(m.group(1)+'.'+volnum)
            
            if not outfiles:
                # Use stem of the source file
                spath = str(s)
                if not spath.endswith('.xml'):
                    outfiles.append(spath+'.'+volnum)
                else:
                    stem, ext = os.path.splitext(spath)
                    outfiles.append(stem+'.'+volnum)
        else:
            # We have to completely rely on the given target name
            outfiles.append(t)
            
        __builder.__call__(env, outfiles[0], s, **kw)
        env.Depends(outfiles[0], kw['DOCBOOK_XSL'])
        result.append(outfiles[0])
        if len(outfiles) > 1:
            env.Clean(outfiles[0], outfiles[1:])

        
    return result