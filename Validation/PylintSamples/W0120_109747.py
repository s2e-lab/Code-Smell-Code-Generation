def find_build_map(stack_builders):
    """
    Find the BUILD_MAP instruction for which the last element of
    ``stack_builders`` is a store.
    """
    assert isinstance(stack_builders[-1], instrs.STORE_MAP)

    to_consume = 0
    for instr in reversed(stack_builders):
        if isinstance(instr, instrs.STORE_MAP):
            # NOTE: This branch should always be hit on the first iteration.
            to_consume += 1
        elif isinstance(instr, instrs.BUILD_MAP):
            to_consume -= instr.arg
            if to_consume <= 0:
                return instr
    else:
        raise DecompilationError(
            "Couldn't find BUILD_MAP for last element of %s." % stack_builders
        )