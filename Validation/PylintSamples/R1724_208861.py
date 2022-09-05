def __word2art(word, font, chr_ignore, letters):
    """
    Return art word.

    :param word: input word
    :type word: str
    :param font: input font
    :type font: str
    :param chr_ignore: ignore not supported character
    :type chr_ignore: bool
    :param letters: font letters table
    :type letters: dict
    :return: ascii art as str
    """
    split_list = []
    result_list = []
    splitter = "\n"
    for i in word:
        if (ord(i) == 9) or (ord(i) == 32 and font == "block"):
            continue
        if (i not in letters.keys()):
            if (chr_ignore):
                continue
            else:
                raise artError(str(i) + " is invalid.")
        if len(letters[i]) == 0:
            continue
        split_list.append(letters[i].split("\n"))
    if font in ["mirror", "mirror_flip"]:
        split_list.reverse()
    if len(split_list) == 0:
        return ""
    for i in range(len(split_list[0])):
        temp = ""
        for j in range(len(split_list)):
            if j > 0 and (
                    i == 1 or i == len(
                        split_list[0]) -
                    2) and font == "block":
                temp = temp + " "
            temp = temp + split_list[j][i]
        result_list.append(temp)
    if "win32" != sys.platform:
        splitter = "\r\n"
    result = (splitter).join(result_list)
    if result[-1] != "\n":
        result += splitter
    return result