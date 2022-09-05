def expectation(cls, method_arg_names):
        """Manages configuration and running of expectation objects.

        Expectation builds and saves a new expectation configuration to the DataAsset object. It is the core decorator \
        used by great expectations to manage expectation configurations.

        Args:
            method_arg_names (List) : An ordered list of the arguments used by the method implementing the expectation \
                (typically the result of inspection). Positional arguments are explicitly mapped to \
                keyword arguments when the expectation is run.

        Notes:
            Intermediate decorators that call the core @expectation decorator will most likely need to pass their \
            decorated methods' signature up to the expectation decorator. For example, the MetaPandasDataset \
            column_map_expectation decorator relies on the DataAsset expectation decorator, but will pass through the \
            signature from the implementing method.

            @expectation intercepts and takes action based on the following parameters:
                * include_config (boolean or None) : \
                    If True, then include the generated expectation config as part of the result object. \
                    For more detail, see :ref:`include_config`.
                * catch_exceptions (boolean or None) : \
                    If True, then catch exceptions and include them as part of the result object. \
                    For more detail, see :ref:`catch_exceptions`.
                * result_format (str or None) : \
                    Which output mode to use: `BOOLEAN_ONLY`, `BASIC`, `COMPLETE`, or `SUMMARY`.
                    For more detail, see :ref:`result_format <result_format>`.
                * meta (dict or None): \
                    A JSON-serializable dictionary (nesting allowed) that will be included in the output without modification. \
                    For more detail, see :ref:`meta`.
        """
        def outer_wrapper(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):

                # Get the name of the method
                method_name = func.__name__

                # Combine all arguments into a single new "kwargs"
                all_args = dict(zip(method_arg_names, args))
                all_args.update(kwargs)

                # Unpack display parameters; remove them from all_args if appropriate
                if "include_config" in kwargs:
                    include_config = kwargs["include_config"]
                    del all_args["include_config"]
                else:
                    include_config = self.default_expectation_args["include_config"]

                if "catch_exceptions" in kwargs:
                    catch_exceptions = kwargs["catch_exceptions"]
                    del all_args["catch_exceptions"]
                else:
                    catch_exceptions = self.default_expectation_args["catch_exceptions"]

                if "result_format" in kwargs:
                    result_format = kwargs["result_format"]
                else:
                    result_format = self.default_expectation_args["result_format"]

                # Extract the meta object for use as a top-level expectation_config holder
                if "meta" in kwargs:
                    meta = kwargs["meta"]
                    del all_args["meta"]
                else:
                    meta = None

                # Get the signature of the inner wrapper:
                if PY3:
                    argspec = inspect.getfullargspec(func)[0][1:]
                else:
                    argspec = inspect.getargspec(func)[0][1:]

                if "result_format" in argspec:
                    all_args["result_format"] = result_format
                else:
                    if "result_format" in all_args:
                        del all_args["result_format"]

                all_args = recursively_convert_to_json_serializable(all_args)

                # Patch in PARAMETER args, and remove locally-supplied arguments
                # This will become the stored config
                expectation_args = copy.deepcopy(all_args)

                if "evaluation_parameters" in self._expectations_config:
                    evaluation_args = self._build_evaluation_parameters(expectation_args,
                                                                        self._expectations_config["evaluation_parameters"])  # This will be passed to the evaluation
                else:
                    evaluation_args = self._build_evaluation_parameters(
                        expectation_args, None)

                # Construct the expectation_config object
                expectation_config = DotDict({
                    "expectation_type": method_name,
                    "kwargs": expectation_args
                })

                # Add meta to our expectation_config
                if meta is not None:
                    expectation_config["meta"] = meta

                raised_exception = False
                exception_traceback = None
                exception_message = None

                # Finally, execute the expectation method itself
                try:
                    return_obj = func(self, **evaluation_args)

                except Exception as err:
                    if catch_exceptions:
                        raised_exception = True
                        exception_traceback = traceback.format_exc()
                        exception_message = str(err)

                        return_obj = {
                            "success": False
                        }

                    else:
                        raise(err)

                # Append the expectation to the config.
                self._append_expectation(expectation_config)

                if include_config:
                    return_obj["expectation_config"] = copy.deepcopy(
                        expectation_config)

                if catch_exceptions:
                    return_obj["exception_info"] = {
                        "raised_exception": raised_exception,
                        "exception_message": exception_message,
                        "exception_traceback": exception_traceback
                    }

                # Add a "success" object to the config
                expectation_config["success_on_last_run"] = return_obj["success"]

                # Add meta to return object
                if meta is not None:
                    return_obj['meta'] = meta


                return_obj = recursively_convert_to_json_serializable(
                    return_obj)
                return return_obj

            return wrapper

        return outer_wrapper