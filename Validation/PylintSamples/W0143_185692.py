def removeOntology(self, ontology):
        """
        Removes the specified ontology term map from this repository.
        """
        q = models.Ontology.delete().where(id == ontology.getId())
        q.execute()