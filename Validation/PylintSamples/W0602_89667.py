def getKnownPlayers(reset=False):
    """identify all of the currently defined players"""
    global playerCache
    if not playerCache or reset:
        jsonFiles = os.path.join(c.PLAYERS_FOLDER, "*.json")
        for playerFilepath in glob.glob(jsonFiles):
            filename = os.path.basename(playerFilepath)
            name = re.sub("^player_", "", filename)
            name = re.sub("\.json$",  "", name)
            player = PlayerRecord(name)
            playerCache[player.name] = player
    return playerCache