def fasta_iter(handle, header=None):
    """Iterate over FASTA file and return FASTA entries

    Args:
        handle (file): FASTA file handle, can be any iterator so long as it
            it returns subsequent "lines" of a FASTA entry

        header (str): Header line of next FASTA entry, if 'handle' has been
            partially read and you want to start iterating at the next entry,
            read the next FASTA header and pass it to this variable when
            calling fasta_iter. See 'Examples.'

    Yields:
        FastaEntry: class containing all FASTA data

    Raises:
        IOError: If FASTA entry doesn't start with '>'

    Examples:
        The following two examples demonstrate how to use fasta_iter.
        Note: These doctests will not pass, examples are only in doctest
        format as per convention. bio_utils uses pytests for testing.

        >>> for entry in fasta_iter(open('test.fasta')):
        ...     print(entry.id)  # Print FASTA id
        ...     print(entry.description)  # Print FASTA description
        ...     print(entry.sequence)  # Print FASTA sequence
        ...     print(entry.write())  # Print full FASTA entry

        >>> fasta_handle = open('test.fasta')
        >>> next(fasta_handle)  # Skip first entry header
        >>> next(fasta_handle)  # Skip first entry sequence
        >>> first_line = next(fasta_handle)  # Read second entry header
        >>> for entry in fasta_iter(fasta_handle, header=first_line):
        ...     print(entry.id)  # Print FASTA id
        ...     print(entry.description)  # Print FASTA description
        ...     print(entry.sequence)  # Print FASTA sequence
        ...     print(entry.write())  # Print full FASTA entry
    """

    # Speed tricks: reduces function calls
    append = list.append
    join = str.join
    strip = str.strip

    next_line = next

    if header is None:
        header = next(handle)  # Read first FASTQ entry header

    # Check if input is text or bytestream
    if (isinstance(header, bytes)):
        def next_line(i):
            return next(i).decode('utf-8')

        header = strip(header.decode('utf-8'))
    else:
        header = strip(header)

    try:  # Manually construct a for loop to improve speed by using 'next'

        while True:  # Loop until StopIteration Exception raised

            line = strip(next_line(handle))

            data = FastaEntry()

            try:
                if not header[0] == '>':
                    raise IOError('Bad FASTA format: no ">" at beginning of line')
            except IndexError:
                raise IOError('Bad FASTA format: file contains blank lines')

            try:
                data.id, data.description = header[1:].split(' ', 1)
            except ValueError:  # No description
                data.id = header[1:]
                data.description = ''

            # Obtain sequence
            sequence_list = []
            while line and not line[0] == '>':
                append(sequence_list, line)
                line = strip(next_line(handle))  # Raises StopIteration at EOF
            header = line  # Store current line so it's not lost next iteration
            data.sequence = join('', sequence_list)

            yield data

    except StopIteration:  # Yield last FASTA entry
        data.sequence = ''.join(sequence_list)
        yield data