def update_appt(self, complex: str, house: str, price: str, square: str, id: str, **kwargs):
        """
        Update existing appartment
        """
        self.check_house(complex, house)

        kwargs['price'] = self._format_decimal(price)
        kwargs['square'] = self._format_decimal(square)

        self.put('developers/{developer}/complexes/{complex}/houses/{house}/appts/{id}'.format(
            developer=self.developer,
            complex=complex,
            house=house,
            id=id,
            price=self._format_decimal(price),
        ), data=kwargs)