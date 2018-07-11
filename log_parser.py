from os import listdir

def main(args):
	outfile = open(args.out_file,  'a')
	files = listdir(args.in_dir)

	for file in files:
		loaded_file = open(args.in_dir + "/" +file,  'r')
		outstr = ""
		first = True
		for line in loaded_file:
			if(first):
				first = False
				if("human" in file):
					outstr += "1,"
				else:
					outstr += "0,"

				out = ""
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


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--in_dir', dest='in_dir')
	parser.add_argument('--out_file', dest='out_file')
	args = parser.parse_args()
	main(args)

