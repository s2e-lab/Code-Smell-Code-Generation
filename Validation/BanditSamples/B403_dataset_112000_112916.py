def loadNetworkFromFile(filename, mode = 'pickle'):
    """
    Deprecated. Use loadNetwork instead.
    """
    if mode == 'pickle':
        import pickle
        fp = open(filename)
        network = pickle.load(fp)
        fp.close()
        return network
    elif mode in ['plain', 'conx']:
        fp = open(filename, "r")
        line = fp.readline()
        network = None
        while line:
            if line.startswith("layer,"):
                # layer, name, size
                temp, name, sizeStr = line.split(",")
                name = name.strip()
                size = int(sizeStr)
                network.addLayer(name, size)
                line = fp.readline()
                weights = [float(f) for f in line.split()]
                for i in range(network[name].size):
                    network[name].weight[i] = weights[i]
            elif line.startswith("connection,"):
                # connection, fromLayer, toLayer
                temp, nameFrom, nameTo = line.split(",")
                nameFrom, nameTo = nameFrom.strip(), nameTo.strip()
                network.connect(nameFrom, nameTo)
                for i in range(network[nameFrom].size):
                    line = fp.readline()
                    weights = [float(f) for f in line.split()]
                    for j in range(network[nameTo].size):
                        network[nameFrom, nameTo].weight[i][j] = weights[j]
            elif line.startswith("parameter,"):
                temp, exp = line.split(",")
                exec(exp) # network is the neural network object
            elif line.startswith("network,"):
                temp, netType = line.split(",")
                netType = netType.strip().lower()
                if netType == "cascornetwork":
                    from pyrobot.brain.cascor import CascorNetwork
                    network = CascorNetwork()
                elif netType == "network":
                    network = Network()
                elif netType == "srn":
                    network = SRN()
                else:
                    raise AttributeError("unknown network type: '%s'" % netType)
            line = fp.readline()
        return network