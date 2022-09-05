def slits_to_ds9_reg(ds9reg, slits):
    """Transform fiber traces to ds9-region format.

    Parameters
    ----------
    ds9reg : BinaryIO
        Handle to output file name in ds9-region format.
    """

    # open output file and insert header

    ds9reg.write('# Region file format: DS9 version 4.1\n')
    ds9reg.write(
        'global color=green dashlist=8 3 width=1 font="helvetica 10 '
        'normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 '
        'move=1 delete=1 include=1 source=1\n')
    ds9reg.write('physical\n')

    for idx, slit in enumerate(slits, 1):
        xpos1, y2, xpos2, y2, xpos2, y1, xpos1, y1 = slit
        xc = 0.5 * (xpos1 + xpos2) + 1
        yc = 0.5 * (y1 + y2) + 1
        xd = (xpos2 - xpos1)
        yd = (y2 - y1)
        ds9reg.write('box({0},{1},{2},{3},0)\n'.format(xc, yc, xd, yd))
        ds9reg.write('# text({0},{1}) color=red text={{{2}}}\n'.format(xpos1 - 5, yc, idx))
        ds9reg.write('# text({0},{1}) color=red text={{{2}}}\n'.format(xpos2 + 5, yc, idx + EMIR_NBARS))