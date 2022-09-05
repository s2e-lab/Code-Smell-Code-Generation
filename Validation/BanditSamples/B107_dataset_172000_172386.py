def get_word_level_vocab(self):
        """Provides word level vocabulary

        Returns
        -------
        Vocab
            Word level vocabulary
        """

        def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
            return list(filter(None, re.split(token_delim + '|' + seq_delim, source_str)))

        return VocabProvider._create_squad_vocab(simple_tokenize, self._dataset)