def list(config=None):
    """
    Lists all hosts from ssh config.
    """
    storm_ = get_storm_instance(config)

    try:
        result = colored('Listing entries:', 'white', attrs=["bold", ]) + "\n\n"
        result_stack = ""
        for host in storm_.list_entries(True):

            if host.get("type") == 'entry':
                if not host.get("host") == "*":
                    result += "    {0} -> {1}@{2}:{3}".format(
                        colored(host["host"], 'green', attrs=["bold", ]),
                        host.get("options").get(
                            "user", get_default("user", storm_.defaults)
                        ),
                        host.get("options").get(
                            "hostname", "[hostname_not_specified]"
                        ),
                        host.get("options").get(
                            "port", get_default("port", storm_.defaults)
                        )
                    )

                    extra = False
                    for key, value in six.iteritems(host.get("options")):

                        if not key in ["user", "hostname", "port"]:
                            if not extra:
                                custom_options = colored(
                                    '\n\t[custom options] ', 'white'
                                )
                                result += " {0}".format(custom_options)
                            extra = True

                            if isinstance(value, collections.Sequence):
                                if isinstance(value, builtins.list):
                                    value = ",".join(value)
                                    
                            result += "{0}={1} ".format(key, value)
                    if extra:
                        result = result[0:-1]

                    result += "\n\n"
                else:
                    result_stack = colored(
                        "   (*) General options: \n", "green", attrs=["bold",]
                    )
                    for key, value in six.iteritems(host.get("options")):
                        if isinstance(value, type([])):
                            result_stack += "\t  {0}: ".format(
                                colored(key, "magenta")
                            )
                            result_stack += ', '.join(value)
                            result_stack += "\n"
                        else:
                            result_stack += "\t  {0}: {1}\n".format(
                                colored(key, "magenta"),
                                value,
                            )
                    result_stack = result_stack[0:-1] + "\n"

        result += result_stack
        print(get_formatted_message(result, ""))
    except Exception as error:
        print(get_formatted_message(str(error), 'error'), file=sys.stderr)
        sys.exit(1)