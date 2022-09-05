def html_error_template():
    """Provides a template that renders a stack trace in an HTML format,
    providing an excerpt of code as well as substituting source template
    filenames, line numbers and code for that of the originating source
    template, as applicable.

    The template's default ``encoding_errors`` value is
    ``'htmlentityreplace'``. The template has two options. With the
    ``full`` option disabled, only a section of an HTML document is
    returned. With the ``css`` option disabled, the default stylesheet
    won't be included.

    """
    import mako.template
    return mako.template.Template(r"""
<%!
    from mako.exceptions import RichTraceback, syntax_highlight,\
            pygments_html_formatter
%>
<%page args="full=True, css=True, error=None, traceback=None"/>
% if full:
<html>
<head>
    <title>Mako Runtime Error</title>
% endif
% if css:
    <style>
        body { font-family:verdana; margin:10px 30px 10px 30px;}
        .stacktrace { margin:5px 5px 5px 5px; }
        .highlight { padding:0px 10px 0px 10px; background-color:#9F9FDF; }
        .nonhighlight { padding:0px; background-color:#DFDFDF; }
        .sample { padding:10px; margin:10px 10px 10px 10px;
                  font-family:monospace; }
        .sampleline { padding:0px 10px 0px 10px; }
        .sourceline { margin:5px 5px 10px 5px; font-family:monospace;}
        .location { font-size:80%; }
        .highlight { white-space:pre; }
        .sampleline { white-space:pre; }

    % if pygments_html_formatter:
        ${pygments_html_formatter.get_style_defs()}
        .linenos { min-width: 2.5em; text-align: right; }
        pre { margin: 0; }
        .syntax-highlighted { padding: 0 10px; }
        .syntax-highlightedtable { border-spacing: 1px; }
        .nonhighlight { border-top: 1px solid #DFDFDF;
                        border-bottom: 1px solid #DFDFDF; }
        .stacktrace .nonhighlight { margin: 5px 15px 10px; }
        .sourceline { margin: 0 0; font-family:monospace; }
        .code { background-color: #F8F8F8; width: 100%; }
        .error .code { background-color: #FFBDBD; }
        .error .syntax-highlighted { background-color: #FFBDBD; }
    % endif

    </style>
% endif
% if full:
</head>
<body>
% endif

<h2>Error !</h2>
<%
    tback = RichTraceback(error=error, traceback=traceback)
    src = tback.source
    line = tback.lineno
    if src:
        lines = src.split('\n')
    else:
        lines = None
%>
<h3>${tback.errorname}: ${tback.message|h}</h3>

% if lines:
    <div class="sample">
    <div class="nonhighlight">
% for index in range(max(0, line-4),min(len(lines), line+5)):
    <%
       if pygments_html_formatter:
           pygments_html_formatter.linenostart = index + 1
    %>
    % if index + 1 == line:
    <%
       if pygments_html_formatter:
           old_cssclass = pygments_html_formatter.cssclass
           pygments_html_formatter.cssclass = 'error ' + old_cssclass
    %>
        ${lines[index] | syntax_highlight(language='mako')}
    <%
       if pygments_html_formatter:
           pygments_html_formatter.cssclass = old_cssclass
    %>
    % else:
        ${lines[index] | syntax_highlight(language='mako')}
    % endif
% endfor
    </div>
    </div>
% endif

<div class="stacktrace">
% for (filename, lineno, function, line) in tback.reverse_traceback:
    <div class="location">${filename}, line ${lineno}:</div>
    <div class="nonhighlight">
    <%
       if pygments_html_formatter:
           pygments_html_formatter.linenostart = lineno
    %>
      <div class="sourceline">${line | syntax_highlight(filename)}</div>
    </div>
% endfor
</div>

% if full:
</body>
</html>
% endif
""", output_encoding=sys.getdefaultencoding(),
        encoding_errors='htmlentityreplace')