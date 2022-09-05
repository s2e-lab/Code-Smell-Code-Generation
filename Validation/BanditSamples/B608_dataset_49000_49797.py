def _get_stream_view_infos(
            self,
            trimmed=False):
        """query the sherlock-catalogues database streamed data tables' metadata
        """
        self.log.debug('starting the ``_get_stream_view_infos`` method')

        sqlQuery = u"""
            SELECT * FROM crossmatch_catalogues.tcs_helper_catalogue_tables_info where legacy_table = 0 and table_name not like "legacy%%"  and table_name like "%%stream" order by number_of_rows desc;
        """ % locals()
        streamInfo = readquery(
            log=self.log,
            sqlQuery=sqlQuery,
            dbConn=self.cataloguesDbConn,
            quiet=False
        )

        if trimmed:
            cleanTable = []
            for r in streamInfo:
                orow = collections.OrderedDict(sorted({}.items()))
                for c in self.basicColumns:
                    if c in r:
                        orow[c] = r[c]
                cleanTable.append(orow)
            streamInfo = cleanTable

        self.log.debug('completed the ``_get_stream_view_infos`` method')
        return streamInfo