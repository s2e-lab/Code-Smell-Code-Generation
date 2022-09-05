def add_input(self, input):
        '''
        Add a single build XML output file to our data.
        '''
        events = xml.dom.pulldom.parse(input)
        context = []
        for (event,node) in events:
            if event == xml.dom.pulldom.START_ELEMENT:
                context.append(node)
                if node.nodeType == xml.dom.Node.ELEMENT_NODE:
                    x_f = self.x_name_(*context)
                    if x_f:
                        events.expandNode(node)
                        # expanding eats the end element, hence walking us out one level
                        context.pop()
                        # call handler
                        (x_f[1])(node)
            elif event == xml.dom.pulldom.END_ELEMENT:
                context.pop()