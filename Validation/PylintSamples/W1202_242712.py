def setup_argparse(self, argument_parser):
    """Set up custom arguments needed for this profile."""
    import glob
    import logging
    import argparse

    def get_fonts(pattern):

      fonts_to_check = []
      # use glob.glob to accept *.ufo

      for fullpath in glob.glob(pattern):
        fullpath_absolute = os.path.abspath(fullpath)
        if fullpath_absolute.lower().endswith(".ufo") and os.path.isdir(
            fullpath_absolute):
          fonts_to_check.append(fullpath)
        else:
          logging.warning(
              ("Skipping '{}' as it does not seem "
               "to be valid UFO source directory.").format(fullpath))
      return fonts_to_check

    class MergeAction(argparse.Action):

      def __call__(self, parser, namespace, values, option_string=None):
        target = [item for l in values for item in l]
        setattr(namespace, self.dest, target)

    argument_parser.add_argument(
        'fonts',
        # To allow optional commands like "-L" to work without other input
        # files:
        nargs='*',
        type=get_fonts,
        action=MergeAction,
        help='font file path(s) to check.'
        ' Wildcards like *.ufo are allowed.')

    return ('fonts',)