def build_kal_scan_band_string(kal_bin, band, args):
    """Return string for CLI invocation of kal, for band scan."""
    option_mapping = {"gain": "-g",
                      "device": "-d",
                      "error": "-e"}
    if not sanity.scan_band_is_valid(band):
        err_txt = "Unsupported band designation: %" % band
        raise ValueError(err_txt)
    base_string = "%s -v -s %s" % (kal_bin, band)
    base_string += options_string_builder(option_mapping, args)
    return(base_string)