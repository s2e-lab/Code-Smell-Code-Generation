def connectMsToNet(Facility_presence=0, ConnectedSubaddress_presence=0,
                   UserUser_presence=0, SsVersionIndicator_presence=0):
    """CONNECT Section 9.3.5.2"""
    a = TpPd(pd=0x3)
    b = MessageType(mesType=0x7)  # 00000111
    packet = a / b
    if Facility_presence is 1:
        c = FacilityHdr(ieiF=0x1C, eightBitF=0x0)
        packet = packet / c
    if ConnectedSubaddress_presence is 1:
        d = ConnectedSubaddressHdr(ieiCS=0x4D, eightBitCS=0x0)
        packet = packet / d
    if UserUser_presence is 1:
        e = UserUserHdr(ieiUU=0x7F, eightBitUU=0x0)
        packet = packet / e
    if SsVersionIndicator_presence is 1:
        f = SsVersionIndicatorHdr(ieiSVI=0x7F, eightBitSVI=0x0)
        packet = packet / f
    return packet