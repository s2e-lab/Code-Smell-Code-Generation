def _ctypes_out(parameter):
    """Returns a parameter variable declaration for an output variable for the specified
    parameter.
    """
    if (parameter.dimension is not None and ":" in parameter.dimension
        and "out" in parameter.direction and ("allocatable" in parameter.modifiers or
                                              "pointer" in parameter.modifiers)):
        if parameter.direction == "(inout)":
            return ("type(C_PTR), intent(inout) :: {}_o".format(parameter.name), True)
        else: #self.direction == "(out)" since that is the only other option.
            return ("type(C_PTR), intent(inout) :: {}_c".format(parameter.name), True)