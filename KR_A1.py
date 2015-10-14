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
	
	
#	if i==100:
#		break
		
f.close()

Record.binaryCount= len(data[0].binaries)
Record.numericalCount= len(data[0].numericals)
Record.categoricalCount= len(data[0].categoricals)

#CREATE test and training data sets
k= 10
random.seed()
random.shuffle(data)
test= np.array(data[:len(data)/k])
train= np.array(data[len(data)/k :])

data = np.array(data) #THIS SHOULD NOT BE REMOVED



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
	
	AttrTypeName=["categorical", "binary", "numerical"]
	
	nodeNum=0	
	
	def __init__(self, level, data):
		self.level=level
		self.data=data
		self.children= []
		self.nodeNum= Node.nodeNum
		Node.nodeNum= Node.nodeNum+1
		self.SetIsLeaf()
		
	def SetIsLeaf(self):
		self.isLeaf = (self.level == 10) or (len(self.data)==4) or (Entropy(self.data)<1.2)
				
	def SetFinalLabel(self):	#Leaf node, take the label that has majority

		if (len(self.data)==0):
			self.finalLabel=1
		else:
			tgts= [ r.tgt.val for r in self.data]
			counts= np.array([ tgts.count(v) for v in Contraception.vals])
			self.finalLabel= np.argmax(counts) + Contraception.No
				
	def CreateTree(self):
		if self.isLeaf:
			self.SetFinalLabel()
		else:
			self.SetBestSplitAttr()
			self.CreateSubTrees()
			
		self.LogCreation()

			
	def LogData(self):
		for r in self.data:
			print r
	
	def LogCreation(self):
		print "Node",self.nodeNum
		
#		self.LogData()
		
		if self.isLeaf:
			print "Node leaf with data entropy=",Entropy(self.data),", no more branching"
			print "Always predict", self.finalLabel
		else:
				
			print "dividing by", Node.AttrTypeName[self.dividingAttrT], "no", self.dividingAttrI
			if self.dividingAttrT== Node.numerical:
				print "threshold=", self.dividingThresh
		
			print "\tChildren",
			for ch in self.children:
				print "Node", ch.nodeNum,
			
			print ""
		

	def Classify(self,r):
		if self.isLeaf:
			return self.finalLabel
		elif self.dividingAttrT== Node.binary:
			if r.binaries[self.dividingAttrI]:
				i=1
			else:
				i=0
		elif self.dividingAttrT== Node.numerical:
			if r.numericals[self.dividingAttrI]< self.dividingThresh:
				i=0
			else:
				i=1
		elif self.dividingAttrT== Node.categorical:
			i=r.categoricals[self.dividingAttrI].val - CatAttr.low
		return self.children[i].Classify(r) 
				
	def CreateSubTrees(self):
		if (self.dividingAttrT== Node.numerical):
			dataDivisions= self.DivideNumerical(self.dividingAttrI, self.dividingThresh)
		else:
			dataDivisions= self.DivideNonNumerical(self.dividingAttrT, self.dividingAttrI) 
			
		self.children= [ Node(self.level+1, d) for d in dataDivisions]
		
		for ch in self.children:
			ch.CreateTree()
		

	def DivideNonNumerical(self, dividingAttrT, dividingAttrI):
		if (dividingAttrT== Node.binary):
			vals= np.array([r.binaries[dividingAttrI] for r in self.data])
			possibleVals= [False,True]
		elif (dividingAttrT== Node.categorical):
			vals= np.array([r.categoricals[dividingAttrI].val for r in self.data])
			possibleVals= CatAttr.vals
		
		valI=[ np.where(vals==val) for val in possibleVals]
	
		return [ self.data[i[0] ] for i in valI]
			
	
	def DivideNumerical(self, dividingAttrI, thresh):
		vals= np.array([r.numericals[dividingAttrI] for r in self.data])
		valI=[ np.where( (vals>=thresh) == truth) for truth in [False, True] ]
		return [self.data[i[0] ] for i in valI]
	
	def SplitEntropyNonNumerical(self, dividingAttrT, dividingAttrI):	
		return self.SplitEntropy(self.DivideNonNumerical(dividingAttrT, dividingAttrI))
		
	def SplitEntropy(self,dats):
		sizeInv= 1.0/ len(self.data)				
		return sum([ sizeInv * Entropy(dat) * len(dat) for dat in dats])
	
		
	def SplitEntropyNumerical(self, dividingAttrI, thresh):
		return self.SplitEntropy(self.DivideNumerical(dividingAttrI, thresh))
		
	def SetBestSplitAttr(self):
		
		#CATEGORICAL SPLIT... JUST REMEMBER Node.categorical=0
		split= np.array([ self.SplitEntropyNonNumerical(Node.categorical, i) for i in xrange(Record.categoricalCount)])
		bestI= np.array([np.argmin(split), 0, 0])
		best= np.array( [split[bestI[0] ], 0, 22220])

		#BINARY SPLIT
		split= np.array([ self.SplitEntropyNonNumerical(Node.binary, i) for i in xrange(Record.binaryCount)])
		bestI[Node.binary]= np.argmin(split)
		best[Node.binary]= split[bestI[Node.binary]]

		#NUMERICAL SPLIT
		regressionThreshes= [ [ self.RegressionThresh(i, t) for t in Contraception.vals ] for i in xrange(Record.numericalCount)]
			
		split= np.array([ np.array([ self.SplitEntropyNumerical(i,regressionThreshes[i][t- Contraception.No]) for t in Contraception.vals ]) for i in xrange(Record.numericalCount)])
		
		BestTgtIforCatI= np.array( [ np.argmin(split[i]) for i in xrange(Record.numericalCount)])
		BestforCatI= np.array( [split[i][BestTgtIforCatI[i]] for i in xrange(Record.numericalCount)] )
		
		bestI[Node.numerical]=np.argmin(BestforCatI)
		best[Node.numerical]= BestforCatI[bestI[Node.numerical]]
		self.dividingThresh= regressionThreshes[bestI[Node.numerical]][BestTgtIforCatI[bestI[Node.numerical]]]
		
		self.dividingTgt= BestTgtIforCatI[bestI[Node.numerical]]
		bestofbest= np.argmin(best)
		self.dividingAttrI= bestI[bestofbest]
		self.dividingAttrT= bestofbest

		
			

	def RegressionThresh(self, dividingAttrI, tgtToIsolate):
		
		vals= [ (r.numericals[dividingAttrI], 1 if r.tgt.val==tgtToIsolate else -1) for r in self.data]
		
		xs= [val[0] for val in vals]

		xySum= sum([val[0]*val[1] for val in vals])
		
		if xySum==.0:
			wInv=999999 #A very large number,just to avoid division by zero
		else:
			wInv= float(sum([x*x for x in xs])) / xySum
				
		nInv= 1.0/len(self.data)
		
		v=(sum(xs)- wInv*sum([val[1] for val in vals]))*nInv
		
		if (min(xs) <= v and max(xs) >= v):
			return v
		else:
			xs.sort()
			return xs[len(xs)/2]	#the median 
	
root= Node(0,train)
root.CreateTree()

correct=0
for r in test:
	prediction=root.Classify(r)
	print "Prediction", prediction, "Class", r.tgt.val
	if (prediction== r.tgt.val):
		correct= correct +1
		
print float(correct)/len(test)

correct2=0
for r in data:
	if root.Classify(r)== r.tgt.val:
		correct2+=1
		
print float(correct2)/len(data)
