import sys


print("Total arguments:", len(sys.argv))

# program input_csv data_sample_size

if len(sys.argv) >1:
	input_csv = sys.argv[1]
	data_sample_size = int(sys.argv[2])
	print("Input file: ", input_csv )
	print("Sample Size:", data_sample_size )
else:
	print("No arguments")
	print("Syntax: program input_csv data_sample_size")