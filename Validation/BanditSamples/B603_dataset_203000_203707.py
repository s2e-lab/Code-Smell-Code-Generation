def sink_info_cb(self, context, sink_info_p, _, __):
        """Updates self.output"""
        if sink_info_p:
            sink_info = sink_info_p.contents
            volume_percent = round(100 * sink_info.volume.values[0] / 0x10000)
            volume_db = pa_sw_volume_to_dB(sink_info.volume.values[0])
            self.currently_muted = sink_info.mute

            if volume_db == float('-Infinity'):
                volume_db = "-∞"
            else:
                volume_db = int(volume_db)

            muted = self.muted if sink_info.mute else self.unmuted

            if self.multi_colors and not sink_info.mute:
                color = self.get_gradient(volume_percent, self.colors)
            else:
                color = self.color_muted if sink_info.mute else self.color_unmuted

            if muted and self.format_muted is not None:
                output_format = self.format_muted
            else:
                output_format = self.format

            if self.bar_type == 'vertical':
                volume_bar = make_vertical_bar(volume_percent, self.vertical_bar_width)
            elif self.bar_type == 'horizontal':
                volume_bar = make_bar(volume_percent)
            else:
                raise Exception("bar_type must be 'vertical' or 'horizontal'")

            selected = ""
            dump = subprocess.check_output("pacmd dump".split(), universal_newlines=True)
            for line in dump.split("\n"):
                if line.startswith("set-default-sink"):
                    default_sink = line.split()[1]
                    if default_sink == self.current_sink:
                        selected = self.format_selected

            self.output = {
                "color": color,
                "full_text": output_format.format(
                    muted=muted,
                    volume=volume_percent,
                    db=volume_db,
                    volume_bar=volume_bar,
                    selected=selected),
            }

            self.send_output()