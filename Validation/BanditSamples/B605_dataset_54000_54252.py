def catCSVs(folder, ouputFileName, removeDups = False) :
	"""Concatenates all csv in 'folder' and wites the results in 'ouputFileName'. My not work on non Unix systems"""
	strCmd = r"""cat %s/*.csv > %s""" %(folder, ouputFileName)
	os.system(strCmd)

	if removeDups :
		removeDuplicates(ouputFileName, ouputFileName)