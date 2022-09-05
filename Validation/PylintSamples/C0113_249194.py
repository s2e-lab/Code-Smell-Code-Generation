def expand_dimension(self, newdim, dimension, maps={}, relations={}):
        ''' When we expand we need to provide new maps and relations as those
        can't be inferred '''
        
        for name, attr in self.__attributes__.items():
            if attr.dim == dimension:
                newattr = attr.copy()
                newattr.empty(newdim - attr.size)
                self.__attributes__[name] = concatenate_attributes([attr, newattr])

        for name, rel in self.__relations__.items():
            if dimension == rel.dim:
                # We need the new relation from the user
                if not rel.name in relations:
                    raise ValueError('You need to provide the relation {} for this resize'.format(rel.name))
                else:
                    if len(relations[name]) != newdim:
                        raise ValueError('New relation {} should be of size {}'.format(rel.name, newdim))
                    else:
                        self.__relations__[name].value = relations[name]
            
            elif dimension == rel.map:
                # Extend the index
                rel.index = range(newdim)
                
        for (a, b), rel in self.maps.items():
            if dimension == rel.dim:
                # We need the new relation from the user
                if not (a, b) in maps:
                    raise ValueError('You need to provide the map {}->{} for this resize'.format(a,  b))
                else:
                    if len(maps[a, b]) != newdim:
                        raise ValueError('New map {} should be of size {}'.format(rel.name, newdim))
                    else:
                        rel.value = maps[a, b]
            
            elif dimension == rel.map:
                # Extend the index
                rel.index = range(newdim)
        
        # Update dimensions
        self.dimensions[dimension] = newdim
        
        return self