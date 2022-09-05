def generate_index_file(filename):
    """Constructs a default home page for the project."""
    with open(filename, 'w') as file:
        content = open(os.path.join(os.path.dirname(__file__), 'templates/index_page.html'), 'r').read()
        file.write(content)