def releaseCompleteNetToMs(Cause_presence=0, Facility_presence=0,
                           UserUser_presence=0):
    """RELEASE COMPLETE Section 9.3.19.1"""
    a = TpPd(pd=0x3)
    b = MessageType(mesType=0x2a)  # 00101010
    packet = a / b
    if Cause_presence is 1:
        c = CauseHdr(ieiC=0x08, eightBitC=0x0)
        packet = packet / c
    if Facility_presence is 1:
        d = FacilityHdr(ieiF=0x1C, eightBitF=0x0)
        packet = packet / d
    if UserUser_presence is 1:
        e = UserUserHdr(ieiUU=0x7E, eightBitUU=0x0)
        packet = packet / e
    return packet