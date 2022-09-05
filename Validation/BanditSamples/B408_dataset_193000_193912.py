def SVGdocument():
	"Create default SVG document"

	import xml.dom.minidom
	implementation = xml.dom.minidom.getDOMImplementation()
	doctype = implementation.createDocumentType(
		"svg", "-//W3C//DTD SVG 1.1//EN",
		"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"
	)
	document= implementation.createDocument(None, "svg", doctype)
	document.documentElement.setAttribute(
		'xmlns', 'http://www.w3.org/2000/svg'
	)
	return document