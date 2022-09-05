def cmd_startstop(options):
    """Start or Stop the specified instance.

    Finds instances that match args and instance-state expected by the
    command.  Then, the target instance is determined, the action is
    performed on the instance, and the eturn information is displayed.

    Args:
        options (object): contains args and data from parser.

    """
    statelu = {"start": "stopped", "stop": "running"}
    options.inst_state = statelu[options.command]
    debg.dprint("toggle set state: ", options.inst_state)
    (i_info, param_str) = gather_data(options)
    (tar_inst, tar_idx) = determine_inst(i_info, param_str, options.command)
    response = awsc.startstop(tar_inst, options.command)
    responselu = {"start": "StartingInstances", "stop": "StoppingInstances"}
    filt = responselu[options.command]
    resp = {}
    state_term = ('CurrentState', 'PreviousState')
    for i, j in enumerate(state_term):
        resp[i] = response["{0}".format(filt)][0]["{0}".format(j)]['Name']
    print("Current State: {}{}{}  -  Previous State: {}{}{}\n".
          format(C_STAT[resp[0]], resp[0], C_NORM,
                 C_STAT[resp[1]], resp[1], C_NORM))