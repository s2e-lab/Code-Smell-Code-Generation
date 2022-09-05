def purge(self):
    """ Purge cache by removing obsolete items. """
    purged_count = 0
    if self.__expiration is not None:
      with self.__connection:
        if self.__caching_strategy is CachingStrategy.FIFO:
          # dump least recently added rows
          for post in (False, True):
            purged_count += self.__connection.execute("DELETE FROM " +
                                                      self.getDbTableName(post=post) + " "
                                                      "WHERE (strftime('%s', 'now') - added_timestamp) > ?;",
                                                      (self.__expiration,)).rowcount
        elif self.__caching_strategy is CachingStrategy.LRU:
          # dump least recently accessed rows
          for post in (False, True):
            purged_count += self.__connection.execute("DELETE FROM " +
                                                      self.getDbTableName(post=post) + " "
                                                      "WHERE (strftime('%s', 'now') - last_accessed_timestamp) > ?;",
                                                      (self.__expiration,)).rowcount
    return purged_count