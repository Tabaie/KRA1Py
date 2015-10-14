import random
f= open("cmc.data","rt")

data= []

fs= f.readline()

i=0
while not fs==None and not fs=="":
	i=i+1
	data.append(fs)
	fs=f.readline()
	
random.seed()
random.shuffle(data)

for s in data:
	print s,
