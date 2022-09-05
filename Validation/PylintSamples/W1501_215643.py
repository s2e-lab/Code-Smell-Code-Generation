def predict(self, X, class_name=None, method_name=None, tnp_dir='tmp',
                keep_tmp_dir=False, num_format=lambda x: str(x)):
        """
        Predict using the transpiled model.

        Parameters
        ----------
        :param X : {array-like}, shape (n_features) or (n_samples, n_features)
            The input data.

        :param class_name : string, default: None
            The name for the ported class.

        :param method_name : string, default: None
            The name for the ported method.

        :param tnp_dir : string, default: 'tmp'
            The path to the temporary directory for
            storing the transpiled (and compiled) model.

        :param keep_tmp_dir : bool, default: False
            Whether to delete the temporary directory
            or not.

        :param num_format : lambda x, default: lambda x: str(x)
            The representation of the floating-point values.

        Returns
        -------
            y : int or array-like, shape (n_samples,)
            The predicted class or classes.
        """

        if class_name is None:
            class_name = self.estimator_name

        if method_name is None:
            method_name = self.target_method

        # Dependencies:
        if not self._tested_dependencies:
            self._test_dependencies()
            self._tested_dependencies = True

        # Support:
        if 'predict' not in set(self.template.SUPPORTED_METHODS):
            error = "Currently the given model method" \
                    " '{}' isn't supported.".format('predict')
            raise AttributeError(error)

        # Cleanup:
        Shell.call('rm -rf {}'.format(tnp_dir))
        Shell.call('mkdir {}'.format(tnp_dir))

        # Transpiled model:
        details = self.export(class_name=class_name,
                              method_name=method_name,
                              num_format=num_format,
                              details=True)
        filename = Porter._get_filename(class_name, self.target_language)
        target_file = os.path.join(tnp_dir, filename)
        with open(target_file, str('w')) as file_:
            file_.write(details.get('estimator'))

        # Compilation command:
        comp_cmd = details.get('cmd').get('compilation')
        if comp_cmd is not None:
            Shell.call(comp_cmd, cwd=tnp_dir)

        # Execution command:
        exec_cmd = details.get('cmd').get('execution')
        exec_cmd = str(exec_cmd).split()

        pred_y = None

        # Single feature set:
        if exec_cmd is not None and len(X.shape) == 1:
            full_exec_cmd = exec_cmd + [str(sample).strip() for sample in X]
            pred_y = Shell.check_output(full_exec_cmd, cwd=tnp_dir)
            pred_y = int(pred_y)

        # Multiple feature sets:
        if exec_cmd is not None and len(X.shape) > 1:
            pred_y = np.empty(X.shape[0], dtype=int)
            for idx, features in enumerate(X):
                full_exec_cmd = exec_cmd + [str(f).strip() for f in features]
                pred = Shell.check_output(full_exec_cmd, cwd=tnp_dir)
                pred_y[idx] = int(pred)

        # Cleanup:
        if not keep_tmp_dir:
            Shell.call('rm -rf {}'.format(tnp_dir))

        return pred_y