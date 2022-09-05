def blend(self, cycles=1):
        """
        Explands the existing Palette by inserting the blending colour
        between all Colours already in the Palette.

        Changes the Palette in-place.

        args:
            cycles(int): number of *blend* cycles to apply. (Default is 1)

        Example usage:

        .. code-block:: python

            p1.blend()
            p1.to_image('p1_blended.png', 60, vertical=False)

        .. image:: p1_blended.png

        .. code-block:: python

            p2.blend()
            p2.to_image('p2_blended.png', 60, vertical=False)

        .. image:: p2_blended.png

        The *blend* functionallity can be applied several times in a sequence
        by use of the *cycles* parameter. This may be useful to quickly get a
        longer series of intermediate colours.

        .. code-block:: python

            p3 = Palette(Colour('#fff'), Colour('#7e1e9c'))
            p3.blend(cycles=5)
            p3.to_image('p3.png', max_width=360, vertical=False)

        .. image:: p3.png

        .. seealso:: :py:func:`colourettu.blend`
        """

        for j in range(int(cycles)):
            new_colours = []
            for i, c in enumerate(self._colours):
                if i != 0:
                    c2 = blend(c, self._colours[i-1])
                    new_colours.append(c2)
                new_colours.append(c)

            self._colours = new_colours