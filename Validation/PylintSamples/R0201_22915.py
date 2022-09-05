def DEFINE_alias(name, original_name, flag_values=FLAGS, module_name=None):  # pylint: disable=g-bad-name
  """Defines an alias flag for an existing one.

  Args:
    name: A string, name of the alias flag.
    original_name: A string, name of the original flag.
    flag_values: FlagValues object with which the flag will be registered.
    module_name: A string, the name of the module that defines this flag.

  Raises:
    gflags.FlagError:
      UnrecognizedFlagError: if the referenced flag doesn't exist.
      DuplicateFlagError: if the alias name has been used by some existing flag.
  """
  if original_name not in flag_values:
    raise UnrecognizedFlagError(original_name)
  flag = flag_values[original_name]

  class _Parser(ArgumentParser):
    """The parser for the alias flag calls the original flag parser."""

    def parse(self, argument):
      flag.parse(argument)
      return flag.value

  class _FlagAlias(Flag):
    """Overrides Flag class so alias value is copy of original flag value."""

    @property
    def value(self):
      return flag.value

    @value.setter
    def value(self, value):
      flag.value = value

  help_msg = 'Alias for --%s.' % flag.name
  # If alias_name has been used, gflags.DuplicatedFlag will be raised.
  DEFINE_flag(_FlagAlias(_Parser(), flag.serializer, name, flag.default,
                         help_msg, boolean=flag.boolean),
              flag_values, module_name)