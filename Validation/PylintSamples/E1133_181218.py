def auto_batch_commands(commands):
    """Given a pipeline of commands this attempts to merge the commands
    into more efficient ones if that is possible.
    """
    pending_batch = None

    for command_name, args, options, promise in commands:
        # This command cannot be batched, return it as such.
        if command_name not in AUTO_BATCH_COMMANDS:
            if pending_batch:
                yield merge_batch(*pending_batch)
                pending_batch = None
            yield command_name, args, options, promise
            continue

        assert not options, 'batch commands cannot merge options'
        if pending_batch and pending_batch[0] == command_name:
            pending_batch[1].append((args, promise))
        else:
            if pending_batch:
                yield merge_batch(*pending_batch)
            pending_batch = (command_name, [(args, promise)])

    if pending_batch:
        yield merge_batch(*pending_batch)