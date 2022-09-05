def _delete(self, paths: Iterable[str]) -> None:
        """
        Delete a collection of paths from S3.

        :param paths: The paths to delete. The prefix will be prepended to each
                      one.
        :raises ClientError: If any request fails.
        """
        for chunk in util.chunk(paths, self._MAX_DELETES_PER_REQUEST):
            keys = list([self._prefix + key for key in chunk])
            logger.info('Deleting %d objects (%s)', len(keys), ', '.join(keys))
            response = self._bucket.delete_objects(Delete={
                'Objects': [{'Key': key} for key in keys],
                'Quiet': True
            })
            logger.debug('Delete objects response: %s', response)