def update(self, **data):
        """
        Update records in the table with +data+. Often combined with `where`,
        as it acts on all records in the table unless restricted.

        ex)

        >>> Repo("foos").update(name="bar")
        UPDATE foos SET name = "bar"
        """
        data = data.items()
        update_command_arg = ", ".join("{} = ?".format(entry[0])
                                       for entry in data)
        cmd = "update {table} set {update_command_arg} {where_clause}".format(
            update_command_arg=update_command_arg,
            where_clause=self.where_clause,
            table=self.table_name).rstrip()
        Repo.db.execute(cmd, [entry[1] for entry in data] + self.where_values)