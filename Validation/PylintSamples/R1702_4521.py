def _read(self, directory, filename, session, path, name, extension, spatial, spatialReferenceID, replaceParamFile):
        """
        Link Node Dataset File Read from File Method
        """
        # Set file extension property
        self.fileExtension = extension

        # Dictionary of keywords/cards and parse function names
        KEYWORDS = ('NUM_LINKS',
                    'TIME_STEP',
                    'NUM_TS',
                    'START_TIME',
                    'TS')

        # Parse file into chunks associated with keywords/cards
        with open(path, 'r') as f:
            self.name = f.readline().strip()
            chunks = pt.chunk(KEYWORDS, f)

        # Parse chunks associated with each key
        for card, chunkList in iteritems(chunks):
            # Parse each chunk in the chunk list
            for chunk in chunkList:
                schunk = chunk[0].strip().split()

                # Cases
                if card == 'NUM_LINKS':
                    # NUM_LINKS handler
                    self.numLinks = schunk[1]

                elif card == 'TIME_STEP':
                    # TIME_STEP handler
                    self.timeStepInterval = schunk[1]

                elif card == 'NUM_TS':
                    # NUM_TS handler
                    self.numTimeSteps = schunk[1]

                elif card == 'START_TIME':
                    # START_TIME handler
                    self.startTime = '%s  %s    %s  %s  %s  %s' % (
                        schunk[1],
                        schunk[2],
                        schunk[3],
                        schunk[4],
                        schunk[5],
                        schunk[6])

                elif card == 'TS':
                    # TS handler
                    for line in chunk:
                        sline = line.strip().split()
                        token = sline[0]

                        # Cases
                        if token == 'TS':
                            # Time Step line handler
                            timeStep = LinkNodeTimeStep(timeStep=sline[1])
                            timeStep.linkNodeDataset = self

                        else:
                            # Split the line
                            spLinkLine = line.strip().split()

                            # Create LinkDataset GSSHAPY object
                            linkDataset = LinkDataset()
                            linkDataset.numNodeDatasets = int(spLinkLine[0])
                            linkDataset.timeStep = timeStep
                            linkDataset.linkNodeDatasetFile = self

                            # Parse line into NodeDatasets
                            NODE_VALUE_INCREMENT = 2
                            statusIndex = 1
                            valueIndex = statusIndex + 1

                            # Parse line into node datasets
                            if linkDataset.numNodeDatasets > 0:
                                for i in range(0, linkDataset.numNodeDatasets):
                                    # Create NodeDataset GSSHAPY object
                                    nodeDataset = NodeDataset()
                                    nodeDataset.status = int(spLinkLine[statusIndex])
                                    nodeDataset.value = float(spLinkLine[valueIndex])
                                    nodeDataset.linkDataset = linkDataset
                                    nodeDataset.linkNodeDatasetFile = self

                                    # Increment to next status/value pair
                                    statusIndex += NODE_VALUE_INCREMENT
                                    valueIndex += NODE_VALUE_INCREMENT
                            else:
                                nodeDataset = NodeDataset()
                                nodeDataset.value = float(spLinkLine[1])
                                nodeDataset.linkDataset = linkDataset
                                nodeDataset.linkNodeDatasetFile = self