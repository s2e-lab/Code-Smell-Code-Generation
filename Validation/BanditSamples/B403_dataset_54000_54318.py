def save(self, filename):
        """
        Save model to pickle file. External feature function is not stored
        """
        import dill

        tmpmodelparams = self.modelparams.copy()
        # fv_extern_src = None
        fv_extern_name = None
        # try:
        #     fv_extern_src = dill.source.getsource(tmpmodelparams['fv_extern'])
        #     tmpmodelparams.pop('fv_extern')
        # except:
        #     pass

        # fv_extern_name = dill.source.getname(tmpmodelparams['fv_extern'])
        if "fv_extern" in tmpmodelparams:
            tmpmodelparams.pop("fv_extern")

        sv = {
            "modelparams": tmpmodelparams,
            "mdl": self.mdl,
            # 'fv_extern_src': fv_extern_src,
            # 'fv_extern_src_name': fv_extern_src_name,
            # 'fv_extern_name': fv_extern_src_name,
            #
        }
        sss = dill.dumps(self.modelparams)
        logger.debug("pickled " + str(sss))

        dill.dump(sv, open(filename, "wb"))