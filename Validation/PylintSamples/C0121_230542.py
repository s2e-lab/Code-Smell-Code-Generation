def mavlink_packet(self, m):
        '''handle and incoming mavlink packet'''
        if m.get_type() == "FENCE_STATUS":
            self.last_fence_breach = m.breach_time
            self.last_fence_status = m.breach_status
        elif m.get_type() in ['SYS_STATUS']:
            bits = mavutil.mavlink.MAV_SYS_STATUS_GEOFENCE

            present = ((m.onboard_control_sensors_present & bits) == bits)
            if self.present == False and present == True:
                self.say("fence present")
            elif self.present == True and present == False:
                self.say("fence removed")
            self.present = present

            enabled = ((m.onboard_control_sensors_enabled & bits) == bits)
            if self.enabled == False and enabled == True:
                self.say("fence enabled")
            elif self.enabled == True and enabled == False:
                self.say("fence disabled")
            self.enabled = enabled

            healthy = ((m.onboard_control_sensors_health & bits) == bits)
            if self.healthy == False and healthy == True:
                self.say("fence OK")
            elif self.healthy == True and healthy == False:
                self.say("fence breach")
            self.healthy = healthy

            #console output for fence:
            if not self.present:
                self.console.set_status('Fence', 'FEN', row=0, fg='black')
            elif self.enabled == False:
                self.console.set_status('Fence', 'FEN', row=0, fg='grey')
            elif self.enabled == True and self.healthy == True:
                self.console.set_status('Fence', 'FEN', row=0, fg='green')
            elif self.enabled == True and self.healthy == False:
                self.console.set_status('Fence', 'FEN', row=0, fg='red')