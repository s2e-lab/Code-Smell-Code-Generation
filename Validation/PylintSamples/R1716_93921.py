def _check_years(self, cell, prior_year):
        '''
        Helper method which defines the rules for checking for existence of a year indicator. If the
        cell is blank then prior_year is used to determine validity.
        '''
        # Anything outside these values shouldn't auto
        # categorize to strings
        min_year = 1900
        max_year = 2100

        # Empty cells could represent the prior cell's title,
        # but an empty cell before we find a year is not a title
        if is_empty_cell(cell):
            return bool(prior_year)
        # Check if we have a numbered cell between min and max years
        return is_num_cell(cell) and cell > min_year and cell < max_year