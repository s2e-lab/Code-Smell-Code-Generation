def handle_response(self, msg, address):
        """Deal with incoming response packets.  All answers
        are held in the cache, and listeners are notified."""
        now = current_time_millis()

        sigs = []
        precache = []

        for record in msg.answers:
            if isinstance(record, DNSSignature):
                sigs.append(record)
            else:
                precache.append(record)

            for e in precache:
                for s in sigs:
                    if self.verify(e, s):
                        # print "DNS: %s verified with %s" % (e,s)

                        if self.adaptive and e.type == _TYPE_A:
                            if e.address == '\x00\x00\x00\x00':
                                e.address = socket.inet_aton(address)

                        if e in self.cache.entries():
                            if e.is_expired(now):
                                for i in self.hooks:
                                    try:
                                        i.remove(e)
                                    except:
                                        pass
                                self.cache.remove(e)
                                self.cache.remove(s)
                            else:
                                entry = self.cache.get(e)
                                sig = self.cache.get(s)
                                if (entry is not None) and (sig is not None):
                                    for i in self.hooks:
                                        try:
                                            i.update(e)
                                        except:
                                            pass
                                    entry.reset_ttl(e)
                                    sig.reset_ttl(s)
                        else:
                            e.rrsig = s
                            self.cache.add(e)
                            self.cache.add(s)
                            for i in self.hooks:
                                try:
                                    i.add(e)
                                except:
                                    pass

                        precache.remove(e)
                        sigs.remove(s)
                        self.update_record(now, record)

        if self.bypass:
            for e in precache:
                if e in self.cache.entries():
                    if e.is_expired(now):
                        for i in self.hooks:
                            try:
                                i.remove(e)
                            except:
                                pass
                        self.cache.remove(e)
                    else:
                        entry = self.cache.get(e)
                        if (entry is not None):
                            for i in self.hooks:
                                try:
                                    i.update(e)
                                except:
                                    pass
                            entry.reset_ttl(e)
                else:
                    self.cache.add(e)
                    for i in self.hooks:
                        try:
                            i.add(e)
                        except:
                            pass

                self.update_record(now, record)