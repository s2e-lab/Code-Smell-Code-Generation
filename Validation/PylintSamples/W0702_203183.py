def restore(self):
        """
        restore the saved game score and data
        """

        size = self.board.SIZE

        try:
            with open(self.store_file, 'r') as f:
                lines = f.readlines()
                score_str = lines[0]
                self.score = int(lines[1])
        except:
            return False

        score_str_list = score_str.split(' ')
        count = 0

        for i in range(size):
            for j in range(size):
                value = score_str_list[count]
                self.board.setCell(j, i, int(value))
                count += 1

        return True