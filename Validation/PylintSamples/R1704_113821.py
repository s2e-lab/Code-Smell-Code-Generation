def search_regexp(self, pattern, audio_basename=None):
        """
        First joins the words of the word_blocks of timestamps with space, per
        audio_basename. Then matches `pattern` and calculates the index of the
        word_block where the first and last word of the matched result appears
        in. Then presents the output like `search_all` method.

        Note that the leading and trailing spaces from the matched results
        would be removed while determining which word_block they belong to.

        Parameters
        ----------
        pattern : str
            A regex pattern.
        audio_basename : str, optional
            Search only within the given audio_basename.

            Default is `False`.

        Returns
        -------
        search_results : {str: {str: [(float, float)]}}
            A dictionary whose keys are queries and whose values are
            dictionaries whose keys are all the audiofiles in which the query
            is present and whose values are a list whose elements are 2-tuples
            whose first element is the starting second of the query and whose
            values are the ending second. e.g.
            {"apple": {"fruits.wav" : [(1.1, 1.12)]}}
        """

        def indexes_in_transcript_to_start_end_second(index_tup,
                                                      audio_basename):
            """
            Calculates the word block index by having the beginning and ending
            index of the matched result from the transcription

            Parameters
            ----------
            index_tup : (int, tup)
                index_tup is of the form tuple(index_start, index_end)
            audio_basename : str

            Retrun
            ------
            [float, float]
                The time of the output of the matched result. Derived from two
                separate word blocks belonging to the beginning and the end of
                the index_start and index_end.
            """
            space_indexes = [i for i, x in enumerate(
                transcription[audio_basename]) if x == " "]
            space_indexes.sort(reverse=True)
            index_start, index_end = index_tup
            # re.finditer returns the ending index by one more
            index_end -= 1
            while transcription[audio_basename][index_start] == " ":
                index_start += 1
            while transcription[audio_basename][index_end] == " ":
                index_end -= 1
            block_number_start = 0
            block_number_end = len(space_indexes)
            for block_cursor, space_index in enumerate(space_indexes):
                if index_start > space_index:
                    block_number_start = (len(space_indexes) - block_cursor)
                    break
            for block_cursor, space_index in enumerate(space_indexes):
                if index_end > space_index:
                    block_number_end = (len(space_indexes) - block_cursor)
                    break
            return (timestamps[audio_basename][block_number_start].start,
                    timestamps[audio_basename][block_number_end].end)

        timestamps = self.get_timestamps()
        if audio_basename is not None:
            timestamps = {audio_basename: timestamps[audio_basename]}
        transcription = {
            audio_basename: ' '.join(
                [word_block.word for word_block in timestamps[audio_basename]]
            ) for audio_basename in timestamps}
        match_map = map(
            lambda audio_basename: tuple((
                audio_basename,
                re.finditer(pattern, transcription[audio_basename]))),
            transcription.keys())
        search_results = _PrettyDefaultDict(lambda: _PrettyDefaultDict(list))
        for audio_basename, match_iter in match_map:
            for match in match_iter:
                search_results[match.group()][audio_basename].append(
                    tuple(indexes_in_transcript_to_start_end_second(
                        match.span(), audio_basename)))
        return search_results