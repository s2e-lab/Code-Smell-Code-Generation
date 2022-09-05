def update_channels(self):
        '''update which channels provide input'''
        self.interlock_channel = -1
        self.override_channel = -1
        self.zero_I_channel = -1
        self.no_vtol_channel = -1

        # output channels
        self.rsc_out_channel = 9
        self.fwd_thr_channel = 10

        for ch in range(1,16):
            option = self.get_mav_param("RC%u_OPTION" % ch, 0)
            if option == 32:
                self.interlock_channel = ch;
            elif option == 63:
                self.override_channel = ch;
            elif option == 64:
                self.zero_I_channel = ch;
            elif option == 65:
                self.override_channel = ch;
            elif option == 66:
                self.no_vtol_channel = ch;

            function = self.get_mav_param("SERVO%u_FUNCTION" % ch, 0)
            if function == 32:
                self.rsc_out_channel = ch
            if function == 70:
                self.fwd_thr_channel = ch