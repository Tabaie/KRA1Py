from re import findall
import random
from math import log
import numpy as np

#enums
class CatAttr:
	low= 1
	medl= 2
	medh= 3
	high= 4	
	vals=[1,2,3,4]
	valNames=["low", "medl", "medh", "high"]
	def __init__(self, val):
		self.val= int(val)
		
	def __str__(self):
		return CatAttr.valNames[self.val-1]
	
class Contraception:
	No= 1
	Long= 2
	Short= 3
	vals=[1,2,3]

	valNames=["No", "Long", "Short"]
	
	def __init__(self, val):
		self.val= int(val)
		
	def __str__(self):
		return Contraception.valNames[self.val-1]
		

	
BoolVals=[False, True]

#the record class
class Record:
	binaryCount=0
	numericalCount=0
	categoricalCount=0
	def wAge(self):
		return self.numericals[0]

	def nChild(self):
		return self.numericals[1]

	def wEd(self):
		return self.categoricals[0]

	def hEd(self):
		return self.categoricals[1]

	def hOcc(self):
		return self.categoricals[2]

	def livStd(self):
		return self.categoricals[3]
		
	def wMuslim(self):
		return self.binaries[0]

	def wWorking(self):
		return self.binaries[1]

	def goodMedia(self):
		return self.binaries[2]
	
	def contraception(self):
		return self.tgt
	
	def __init__(self,wAge,wEd,hEd,nChild,wMuslim, wWorking, hOcc, livStd, goodMedia, contraception):
		self.numericals= [wAge, nChild]
		
		self.categoricals= [wEd, hEd, hOcc, livStd]
		
		self.binaries= [wMuslim, wWorking, goodMedia]

		self.tgt= contraception
		
	def __str__(self):
		return str(self.wAge())+' '+ str(self.wEd())+' '+str(self.hEd())+' '+str(self.nChild())+' '+str(self.wMuslim())+' '+str(self.wWorking())+' '+str(self.hOcc())+' '+str(self.livStd())+' '+str(self.goodMedia())+' '+str(self.contraception())
		
#MAIN ROUTINE		

random.seed()

#READ DATA FROM FILE
f= open("cmc.data", "rt")

fs= f.readline()
data= list()

i=0
while not fs==None and not fs=="":

	fl= findall(r'\d+', fs)
	
	data.append(Record(int(fl[0]), CatAttr(int(fl[1])), CatAttr(fl[2]), int(fl[3]), BoolVals[int(fl[4])], BoolVals[int(fl[5])], CatAttr(fl[6]), CatAttr(fl[7]), BoolVals[int(fl[8])], Contraception(fl[9]) ) )
		
	fs=f.readline()
	
	print data[i]
	
	i= i+1	
	
	
	if i==5:
		break
		
f.close()

Record.binaryCount= len(data[0].binaries)
Record.numericalCount= len(data[0].numericals)
Record.categoricalCount= len(data[0].categoricals)

#CREATE test and training data sets
k= 3
random.shuffle(data)
test= np.array(data[:len(data)/k])
train= data[len(data)/k :]

data = np.array(data)

#print [r for r in train]

#COMPUTING ENTROPY OF A DATA SET
def Entropy(data):
	if (len(data)==0):
		return .0

	contraVals= [r.tgt.val for r in data]
	entropy=0.0
	count=0
	dataCountInv= 1.0/len(data)
		
	for tgtVal in Contraception.vals:
	
		count = contraVals.count(tgtVal)
		
		if (count>=1):
			frac= count * dataCountInv
			entropy= entropy + log(frac,2)*frac

	return -entropy


#Decision Nodes

class Node:
	categorical=0
	binary=1
	numerical=2
	
#	dividingAttrT #Categorical, Boolean or Number
#	dividingAttrI #Index among its own type
	
#	parent
	
#	data
	
#	children
	
	def SplitEntropy(self, dividingAttrT, dividingAttrI):
		
		if (dividingAttrT== Node.numerical):
			vals= np.array([r.numericals[dividingAttrI] for r in self.data])
			return 200000000000000000000000
			
		elif (dividingAttrT== Node.binary):
			vals= np.array([1 if r.binaries[dividingAttrI] else 0 for r in self.data])
			possibleVals= [0,1]
		elif (dividingAttrT== Node.categorical):
			vals= np.array([r.categoricals[dividingAttrI].val for r in self.data])
			print vals
			possibleVals= CatAttr.vals
		
		valI=[ np.where(vals==val) for val in possibleVals]
	
		dats= [ self.data[i[0] ] for i in valI]
				
		sizeInv= 1.0/ len(self.data)
				
		return sum([ sizeInv * Entropy(dat) * len(dat) for dat in dats])
		
		
	def SetBestSplitAttr(self):
		
		#CATEGORICAL SPLIT... JUST REMEMBER Node.categorical=0
		split= np.array([ self.SplitEntropy(Node.categorical, i) for i in xrange(Record.categoricalCount)])
		bestI= np.array([np.argmin(split), 0, 0])
		best= np.array( [split[bestI[0] ], 0, 22220])

		#BINARY SPLIT
		split= np.array([ self.SplitEntropy(Node.binary, i) for i in xrange(Record.binaryCount)])
		bestI[Node.binary]= np.argmin(split)
		best[Node.binary]= split[bestI[Node.binary]]

		#NUMERICAL SPLIT
		regressionThreshes= [ [
		
		print bestI
		print best
		
		bestofbest= np.argmin(best)
		
		self.dividingAttrI= bestI[bestofbest]
		self.dividingAttrT= bestofbest
			

	def RegressionThresh(self, dividingAttrI, tgtToIsolate):
		
		vals= [ (r.numericals[dividingAttrI], 1 if r.tgt.val==tgtToIsolate else -1) for r in self.data]
		
		xs= [val[0] for val in vals]
		
		wInv= float(sum([x*x for x in xs])) / sum([val[0]*val[1] for val in vals])	
				
		nInv= 1.0/len(self.data)
		
		v=(sum(xs)- wInv*sum([val[1] for val in vals]))*nInv
		
		if (min(xs) <= v and max(xs) >= v):
			return v
		else:
			xs.sort()
			return xs[len(xs)/2]	#the median 
	
root= Node()
root.data= data
print root.RegressionThresh(0,Contraception.No)
#root.SetBestSplitAttr()
#print root.SplitEntropy(Node.categorical, 0)

