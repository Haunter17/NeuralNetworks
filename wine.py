import csv
data = []
with open('wine.csv', 'rb') as csvfile:
	dReader = csv.reader(csvfile, delimiter=' ')
	for row in dReader:
		sample = row[0].split('\t')
		sample = [eval(num) for num in sample]
		feature, label = sample[:-3], sample[-3:]
		# if label == 1:
		# 	label = [1, 0, 0]
		# elif label == 2:
		# 	label = [0, 1, 0]
		# else:
		# 	label = [0, 0, 1]

		data.append([feature] + [label])
		