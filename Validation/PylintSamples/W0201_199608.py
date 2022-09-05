def get_model_parser(top_rule, comments_model, **kwargs):
    """
    Creates model parser for the given language.
    """

    class TextXModelParser(Parser):
        """
        Parser created from textual textX language description.
        Semantic actions for this parser will construct object
        graph representing model on the given language.
        """

        def __init__(self, *args, **kwargs):
            super(TextXModelParser, self).__init__(*args, **kwargs)

            # By default first rule is starting rule
            # and must be followed by the EOF
            self.parser_model = Sequence(
                nodes=[top_rule, EOF()], rule_name='Model', root=True)
            self.comments_model = comments_model

            # Stack for metaclass instances
            self._inst_stack = []

            # Dict for cross-ref resolving
            # { id(class): { obj.name: obj}}
            self._instances = {}

            # List to keep track of all cross-ref that need to be resolved
            # Contained elements are tuples: (instance, metaattr, cross-ref)
            self._crossrefs = []

        def clone(self):
            """
            Responsibility: create a clone in order to parse a separate file.
            It must be possible that more than one clone exist in parallel,
            without being influenced by other parser clones.

            Returns:
                A clone of this parser
            """
            import copy
            the_clone = copy.copy(self)  # shallow copy

            # create new objects for parse-dependent data
            the_clone._inst_stack = []
            the_clone._instances = {}
            the_clone._crossrefs = []

            # TODO self.memoization = memoization
            the_clone.comments = []
            the_clone.comment_positions = {}
            the_clone.sem_actions = {}

            return the_clone

        def _parse(self):
            try:
                return self.parser_model.parse(self)
            except NoMatch as e:
                line, col = e.parser.pos_to_linecol(e.position)
                raise TextXSyntaxError(message=text(e),
                                       line=line,
                                       col=col,
                                       expected_rules=e.rules)

        def get_model_from_file(self, file_name, encoding, debug,
                                pre_ref_resolution_callback=None,
                                is_main_model=True):
            """
            Creates model from the parse tree from the previous parse call.
            If file_name is given file will be parsed before model
            construction.
            """
            with codecs.open(file_name, 'r', encoding) as f:
                model_str = f.read()

            model = self.get_model_from_str(
                model_str, file_name=file_name, debug=debug,
                pre_ref_resolution_callback=pre_ref_resolution_callback,
                is_main_model=is_main_model, encoding=encoding)

            return model

        def get_model_from_str(self, model_str, file_name=None, debug=None,
                               pre_ref_resolution_callback=None,
                               is_main_model=True, encoding='utf-8'):
            """
            Parses given string and creates model object graph.
            """
            old_debug_state = self.debug

            try:
                if debug is not None:
                    self.debug = debug

                if self.debug:
                    self.dprint("*** PARSING MODEL ***")

                self.parse(model_str, file_name=file_name)
                # Transform parse tree to model. Skip root node which
                # represents the whole file ending in EOF.
                model = parse_tree_to_objgraph(
                    self, self.parse_tree[0], file_name=file_name,
                    pre_ref_resolution_callback=pre_ref_resolution_callback,
                    is_main_model=is_main_model, encoding=encoding)
            finally:
                if debug is not None:
                    self.debug = old_debug_state

            try:
                model._tx_metamodel = self.metamodel
            except AttributeError:
                # model is some primitive python type (e.g. str)
                pass
            return model

    return TextXModelParser(**kwargs)