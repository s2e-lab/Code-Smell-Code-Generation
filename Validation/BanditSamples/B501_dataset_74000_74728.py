def validate(filename, verbose=False):
    """
    Validate file and return JSON result as dictionary.

    "filename" can be a file name or an HTTP URL.
    Return "" if the validator does not return valid JSON.
    Raise OSError if curl command returns an error status.
    """
    # is_css = filename.endswith(".css")

    is_remote = filename.startswith("http://") or filename.startswith(
        "https://")
    with tempfile.TemporaryFile() if is_remote else open(
            filename, "rb") as f:

        if is_remote:
            r = requests.get(filename, verify=False)
            f.write(r.content)
            f.seek(0)

        # if is_css:
        #     cmd = (
        #         "curl -sF \"file=@%s;type=text/css\" -F output=json -F warning=0 %s"
        #         % (quoted_filename, CSS_VALIDATOR_URL))
        #     _ = cmd
        # else:
        r = requests.post(
            HTML_VALIDATOR_URL,
            files={"file": (filename, f, "text/html")},
            data={
                "out": "json",
                "showsource": "yes",
            },
            verify=False)

    return r.json()