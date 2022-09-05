def load_hvjs(cls, logo=False, bokeh_logo=False, mpl_logo=False, plotly_logo=False,
                  JS=True, message='HoloViewsJS successfully loaded.'):
        """
        Displays javascript and CSS to initialize HoloViews widgets.
        """
        import jinja2
        # Evaluate load_notebook.html template with widgetjs code
        if JS:
            widgetjs, widgetcss = Renderer.html_assets(extras=False, backends=[], script=True)
        else:
            widgetjs, widgetcss = '', ''

        # Add classic notebook MIME renderer
        widgetjs += nb_mime_js

        templateLoader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
        jinjaEnv = jinja2.Environment(loader=templateLoader)
        template = jinjaEnv.get_template('load_notebook.html')
        html = template.render({'widgetcss':   widgetcss,
                                'logo':        logo,
                                'bokeh_logo':  bokeh_logo,
                                'mpl_logo':    mpl_logo,
                                'plotly_logo': plotly_logo,
                                'message':     message})
        publish_display_data(data={'text/html': html})

        # Vanilla JS mime type is only consumed by classic notebook
        # Custom mime type is only consumed by JupyterLab
        if JS:
            mimebundle = {
                MIME_TYPES['js']           : widgetjs,
                MIME_TYPES['jlab-hv-load'] : widgetjs
            }
            if os.environ.get('HV_DOC_HTML', False):
                mimebundle = {'text/html': mimebundle_to_html(mimebundle)}
            publish_display_data(data=mimebundle)