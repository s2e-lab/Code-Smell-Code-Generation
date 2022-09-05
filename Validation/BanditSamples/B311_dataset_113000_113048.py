def pick(rest):
	"Pick between a few options"
	question = rest.strip()
	choices = util.splitem(question)
	if len(choices) == 1:
		return "I can't pick if you give me only one choice!"
	else:
		pick = random.choice(choices)
		certainty = random.sample(phrases.certainty_opts, 1)[0]
		return "%s... %s %s" % (pick, certainty, pick)