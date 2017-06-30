import os
import subprocess
output=subprocess.check_output(["arp","-a"])
'''
def getIP():
	output=subprocess.check_output(["ipconfig"])
	
	index=output.index("IPv4 Address. . . . . . . . . . . :")

	
	return output[index+36:index+50]
def SaveIP(data):
	text_file = open("Output1.txt", "w")
	text_file.write(data)
	text_file.close()

def ReadPrevIP():
	text_file2 = open("Output1.txt", "r")
	prev_ip=text_file2.read()
	text_file2.close()
	return prev_ip
'''

print(output)
'''IP_1=getIP()

IP_2=ReadPrevIP()
print(IP_1,IP_2)
if IP_1 in  IP_2:
	print IP_1
SaveIP(IP_1)
'''