def Save(nk):
	data=''
	for i in nk:
		print(i)
		data=data+str(i)
		data=data+','
	data=data+"0\n"
	print(data)
	text_file = open("Output1.csv", "a")
	text_file.write(data)
	text_file.close()

