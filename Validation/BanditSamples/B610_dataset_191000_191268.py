def delete(self, constraint):
        """
        Delete a record from the repository
        """

        results = self._get_repo_filter(Service.objects).extra(where=[constraint['where']],
                                                               params=constraint['values']).all()
        deleted = len(results)
        results.delete()
        return deleted