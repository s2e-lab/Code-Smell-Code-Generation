def save_notebook(self):
        """ Saves the current notebook by
        injecting JavaScript to save to .ipynb file.
        """
        try:
            from IPython.display import display, Javascript
        except ImportError:
            log.warning("Could not import IPython Display Function")
            print("Make sure to save your notebook before sending it to OK!")
            return

        if self.mode == "jupyter":
            display(Javascript('IPython.notebook.save_checkpoint();'))
            display(Javascript('IPython.notebook.save_notebook();'))
        elif self.mode == "jupyterlab":
            display(Javascript('document.querySelector(\'[data-command="docmanager:save"]\').click();'))   
                       
        print('Saving notebook...', end=' ')

        ipynbs = [path for path in self.assignment.src
                  if os.path.splitext(path)[1] == '.ipynb']
        # Wait for first .ipynb to save
        if ipynbs:
            if wait_for_save(ipynbs[0]):
                print("Saved '{}'.".format(ipynbs[0]))
            else:
                log.warning("Timed out waiting for IPython save")
                print("Could not automatically save \'{}\'".format(ipynbs[0]))
                print("Make sure your notebook"
                      " is correctly named and saved before submitting to OK!".format(ipynbs[0]))
                return False                
        else:
            print("No valid file sources found")
        return True