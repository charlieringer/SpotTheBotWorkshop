from os import listdir

def main(args):
	outfile = open(args.out_file,  'a')
	files = listdir(args.in_dir)

	outfile.write("Human/AI, ID, Skill, GameID, LevelID, Seed, Win, Score, Tick\n")

	for file in files:
		loaded_file = open(args.in_dir + "/" +file,  'r')
		outstr = ""
		first = True
		for line in loaded_file:
			if(first):
				first = False
				out = getFormatedFileNameData(file)

				
				for char in line:
					if char == " ":
						outstr += "%s," % (out)
						out = ""
					else:
						out += char
				outstr += "%s," % (out.strip())
			else:
				if "ACTION_NIL" in line:
					outstr += "%i," % (0)
				if "ACTION_LEFT" in line:
					outstr += "%i," % (1)
				if "ACTION_RIGHT" in line:
					outstr += "%i," % (2)
				if "ACTION_UP" in line:
					outstr += "%i," % (3)
				if "ACTION_DOWN" in line:
					outstr += "%i," % (4)
				if "ACTION_USE" in line:
					outstr += "%i," % (5)
		outstr+= "\n"
		outfile.write(outstr)

def getFormatedFileNameData(fileName):
	formattedString = ""
	if("human" in fileName):
		formattedString += "1,"
	else:
		formattedString += "0,"

	gameID = ''
	levelID = ''
	userID = ''
	skill = ''

	parsingID = 0
	for char in fileName:
		if char == '_':
			parsingID+=1
			continue
		if parsingID == 1: gameID+=char
		elif parsingID == 2: levelID+=char
		elif parsingID == 3: userID+=char
		elif parsingID == 4: skill+=char
	return formattedString + userID +"," + skill +  "," + gameID +"," + levelID +","

	
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--in_dir', dest='in_dir')
	parser.add_argument('--out_file', dest='out_file')
	args = parser.parse_args()
	main(args)

