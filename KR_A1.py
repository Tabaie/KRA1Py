#AUTHOR: ARYA POURTABATABAIE
from re import findall
import random
from math import log
import numpy as np

#enums

#This class represents categorical attributes. It is assumed that they can either of the 4 possible values
class CatAttr:
#Common names for values
	low= 1
	medl= 2
	medh= 3
	high= 4	
#List of legal values, for use in loops
	vals=[1,2,3,4]
#Name of legal values in order, for use in printing
	valNames=["low", "medl", "medh", "high"]
	
	def __init__(self, val):
		self.val= int(val)
		
#For use in printing
	def __str__(self):
		return CatAttr.valNames[self.val-1]
		
#This is the target categorical attribute. It hold either of three legal values
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
		

#A list of two possible booleans for convenience
BoolVals=[False, True]

#the record class
class Record:
#Number of attributes of each type
	binaryCount=3
	numericalCount=2
	categoricalCount=4

#Accessing the values of the attributes by their name
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
		
#The constructor
	def __init__(self,wAge,wEd,hEd,nChild,wMuslim, wWorking, hOcc, livStd, goodMedia, contraception):
		self.numericals= [wAge, nChild]
		
		self.categoricals= [wEd, hEd, hOcc, livStd]
		
		self.binaries= [wMuslim, wWorking, goodMedia]

		self.tgt= contraception
		
#For use in printing
	def __str__(self):
		return str(self.wAge())+' '+ str(self.wEd())+' '+str(self.hEd())+' '+str(self.nChild())+' '+str(self.wMuslim())+' '+str(self.wWorking())+' '+str(self.hOcc())+' '+str(self.livStd())+' '+str(self.goodMedia())+' '+str(self.contraception())
		
#MAIN ROUTINE		


#READ DATA FROM FILE
f= open("cmc.data", "rt")

fs= f.readline()
datal= list()

while not fs==None and not fs=="":

	fl= findall(r'\d+', fs)	#Extract the numbers (Non-empty strings of digits)
	
	#Create a new record based on the input
	datal.append(Record(int(fl[0]), CatAttr(int(fl[1])), CatAttr(fl[2]), int(fl[3]), BoolVals[int(fl[4])], BoolVals[int(fl[5])], CatAttr(fl[6]), CatAttr(fl[7]), BoolVals[int(fl[8])], Contraception(fl[9]) ) )
	
	fs=f.readline()
				
f.close()


#COMPUTING ENTROPY OF A DATA SET
def Entropy(data):
	if (len(data)==0): #An empty set is deterministic!
		return .0

	tgtVals= [r.tgt.val for r in data] #Target values
	entropy=0.0
	count=0
	dataCountInv= 1.0/len(data)	#Because multiplication is faster than division!
		
	for tgtVal in Contraception.vals: #for any POSSIBLE value
	
		count = tgtVals.count(tgtVal)	#see how many of those are there
		
		if (count>=1):
			frac= count * dataCountInv
			entropy= entropy + log(frac,2)*frac

	return -entropy


#Decision Nodes

class Node:
#Indexes for how types of attributes are represented
	categorical=0
	binary=1
	numerical=2
	
	AttrTypeName=["categorical", "binary", "numerical"]

#Number of the nodes generated	
	nodeNum=0	
	
	def __init__(self, level, data):
		self.level=level	#Depth of the node
		self.data=data		#It's data set
		self.children= [] #Children
		self.nodeNum= Node.nodeNum #Index of the node -- for debugging purporses
		Node.nodeNum= Node.nodeNum+1
		self.SetIsLeaf()	#To see if it is going to be a leaf node
		
	def SetIsLeaf(self):
		self.isLeaf = (self.level == 20) or (len(self.data)==3) or (Entropy(self.data)<.1 )
		#Too deep of a tree, few data points, and low variation among data points are all good criteria for termination
				
	#Classify a data point based on the tree generated
	def Classify(self,r):
		if self.isLeaf:
			return self.finalLabel
		elif self.dividingAttrT== Node.binary:
			if r.binaries[self.dividingAttrI]:
				i=1 # 1 = True
			else:
				i=0	# 0 = False
		elif self.dividingAttrT== Node.numerical:
			if r.numericals[self.dividingAttrI]< self.dividingThresh:
				i=0 # 0 = lower than threshold
			else:
				i=1 # 1 = higher than threshold
		elif self.dividingAttrT== Node.categorical:
			i=r.categoricals[self.dividingAttrI].val - CatAttr.low

		return self.children[i].Classify(r)	#Found the right child, now classify recursively 


	#Come up with a definite label decision if it is a leaf node
	def SetFinalLabel(self):	#Leaf node, take the label that has majority

		if (len(self.data)==0):
			self.finalLabel= random.randint(1,3) #No other criteria!
		else:
			tgts= [ r.tgt.val for r in self.data] #Find the mode, that appears most
			counts= np.array([ tgts.count(v) for v in Contraception.vals])
			self.finalLabel= np.argmax(counts) + Contraception.No
				
	#Generate the subtree of the node
	def CreateTree(self):
	
		if self.isLeaf:
			self.SetFinalLabel()
		else: #if it is not a leaf node
			self.SetBestSplitAttr() #Figure out how the splitting can be done best

			if (self.dividingAttrT== Node.numerical): #Divide the data according to the regression threshold
				dataDivisions= self.DivideNumerical(self.dividingAttrI, self.dividingThresh)
			else: #Divide it according to the values of the categorical/binary attribute
				dataDivisions= self.DivideNonNumerical(self.dividingAttrT, self.dividingAttrI) 
			
			#For every data division, a child node should be managing it
			self.children= [ Node(self.level+1, d) for d in dataDivisions]
		
			#Recursively create the tree
			for ch in self.children:
				ch.CreateTree()
			
		#self.LogCreation()


			
	def LogData(self):
		for r in self.data:
			print r
	
	def LogCreation(self):
		print "Node",self.nodeNum
		
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
		
				
	#Divide the training dataset for the children of the node based on the division criterion chosen
	def DivideNonNumerical(self, dividingAttrT, dividingAttrI): #Parameters: Attribute Type (Binary, Categorical), the Index of the attribute among its own type

		if (dividingAttrT== Node.binary):
			vals= np.array([r.binaries[dividingAttrI] for r in self.data]) #Array of the corresponding values of the records
			possibleVals= [False,True]

		elif (dividingAttrT== Node.categorical):
			vals= np.array([r.categoricals[dividingAttrI].val for r in self.data])
			possibleVals= CatAttr.vals
		
		valI=[ np.where(vals==val) for val in possibleVals]	#for each possible value, make out the indexes of where in the original dataset they reside
	
		return [ self.data[i[0] ] for i in valI] #Now use those indexes to partition the dataset
			
	#Dividing the training dataset... If the dividing attribute is numerical
	def DivideNumerical(self, dividingAttrI, thresh): #Arguments: Attribute index among the numerical attributes, the threshold resulted from regression
		vals= np.array([r.numericals[dividingAttrI] for r in self.data])
		valI=[ np.where( (vals>=thresh) == truth) for truth in [False, True] ]	#indexes of those lower than the threshold and those higher
		return [self.data[i[0] ] for i in valI]
	
	#Compute the entropy IF WE WERE TO divide the dataset based on this particular attribute
	def SplitEntropyNonNumerical(self, dividingAttrT, dividingAttrI):	
		return self.SplitEntropy(self.DivideNonNumerical(dividingAttrT, dividingAttrI))

	#The entropy of a single branch times its weight, should be summed along all branches		
	def SplitEntropy(self,dats): #Dats is a list of lists
		sizeInv= 1.0/ len(self.data)			
		return sum([ sizeInv * Entropy(dat) * len(dat) for dat in dats])
	
		
	def SetBestSplitAttr(self):
		
		#CATEGORICAL SPLIT... JUST REMEMBER Node.categorical=0
		split= np.array([ self.SplitEntropy(self.DivideNonNumerical(Node.categorical, i)) for i in xrange(Record.categoricalCount)])
		bestI= np.array([np.argmin(split), 0, 0]) #The best categorical split
		best= np.array( [split[bestI[0] ], 0, 22220])

		#BINARY SPLITc
		split= np.array([ self.SplitEntropy(self.DivideNonNumerical(Node.binary, i)) for i in xrange(Record.binaryCount)])
		bestI[Node.binary]= np.argmin(split) #The best binary split
		best[Node.binary]= split[bestI[Node.binary]]

		#NUMERICAL SPLIT
		regressionThreshes= [ [ self.RegressionThresh(i, t) for t in Contraception.vals ] for i in xrange(Record.numericalCount)] #The optimal threshold for a given numerical attribute and a given target values it wants to isolate
			
		split= np.array([ np.array([ self.SplitEntropy(self.DivideNumerical(i,regressionThreshes[i][t- Contraception.No])) for t in Contraception.vals ]) for i in xrange(Record.numericalCount)])
		
		BestTgtIforCatI= np.array( [ np.argmin(split[i]) for i in xrange(Record.numericalCount)]) #For each attribute, the best target's index
		BestforCatI= np.array( [split[i][BestTgtIforCatI[i]] for i in xrange(Record.numericalCount)]) #For each attribute, the best entropy


		bestI[Node.numerical] = np.argmin(BestforCatI)
		best[Node.numerical]= BestforCatI[bestI[Node.numerical]]
		self.dividingThresh= regressionThreshes[bestI[Node.numerical]][BestTgtIforCatI[bestI[Node.numerical]]]
		
		self.dividingTgt= BestTgtIforCatI[bestI[Node.numerical]]
		bestofbest= np.argmin(best)
		self.dividingAttrI= bestI[bestofbest]
		self.dividingAttrT= bestofbest
			
		
	#Threshold of regression with respect to a particular attribute and a particular target value to isolate
	def RegressionThresh(self, dividingAttrI, tgtToIsolate):
		
		vals= [ (r.numericals[dividingAttrI], 1 if r.tgt.val==tgtToIsolate else -1) for r in self.data] # (x,y) pairs
		
		xs= [val[0] for val in vals] #x values
		ys= [val[1] for val in vals] #y values

		xSum= sum(xs)
		ySum= sum(ys)

		xySum= sum([val[0]*val[1] for val in vals])
		
		if xySum==.0:
			wInv=999999 #A very large number,just to avoid division by zero
		else:
			wInv= float(sum([x*x for x in xs]) - xSum*xSum) / (xySum - xSum* ySum)
				
		nInv= 1.0/len(self.data)
		
		v=(xSum- wInv*ySum)*nInv
		
		if (min(xs) <= v and max(xs) >= v): #if it's in range return the threshold computed
			return v
		else:
			xs.sort()
			return xs[len(xs)/2]	#the median 
	
#STATS

def Accuracy(confMat, dataSizeInv):
	return dataSizeInv * sum([confMat[i][i] for i in xrange(len(Contraception.vals))])


def Recall(confMat):
	return [float( confMat[i][i])/ sum( [confMat[i][j] for j in xrange(len(Contraception.vals))] ) for i in xrange(len(Contraception.vals))]
	
def Precision(confMat):
	return [float( confMat[i][i])/ sum( [confMat[j][i] for j in xrange(len(Contraception.vals))] ) for i in xrange(len(Contraception.vals))]

def PrintMetricList(metric):
	for m,name in zip(metric,Contraception.valNames):
		print "\tfor", name, ":", m

def mean(l):
	return sum(l)/len(l)
	
def variance(l):
	return (sum([x*x for x in l]) - mean(l) )/len(l)

def MeanOfEach(lB):
	return [ mean([lB[run][attr] for run in xrange(len(lB))] ) for attr in xrange(len(lB[0])) ]
	return [mean(l) for l in lB]
	
def VarianceOfEach(lB):
	return [ variance([lB[run][attr] for run in xrange(len(lB))] ) for attr in xrange(len(lB[0])) ]
	return [variance(l) for l in lB]

#CREATE test and training data sets
k= 10
random.seed()

testSizeInv= 1.0/(len(datal)/k)
dataSizeInv= 1.0/len(datal)

#Statistics of the metrics
taccuracy=[]
tprecision=[]
trecall=[]
tf1=[]

daccuracy=[]
dprecision=[]
drecall=[]
df1=[]

for run in xrange(k):	#Run k times

	random.shuffle(datal)
	test= np.array(datal[:len(datal)/k])
	train= np.array(datal[len(datal)/k :])

	data = np.array(datal) #THIS SHOULD NOT BE REMOVED	


	root= Node(0,train)
	root.CreateTree()

	confusionMat= [ [0 for v in Contraception.vals] for w in Contraception.vals]	# i= actual, j= predicted over test data only

	for r in test:
		prediction=root.Classify(r)

		confusionMat[r.tgt.val-1][prediction-1]+=1		
				
				
	confusionWhole= [ [0 for v in Contraception.vals] for w in Contraception.vals]	# i= actual, j= predicted over all data
	for r in data:
		prediction=root.Classify(r)
		confusionWhole[r.tgt.val-1][prediction-1]+=1		
		
	
	accuracy=Accuracy(confusionMat, testSizeInv)
	precision=Precision(confusionMat)
	recall=Recall(confusionMat)
	f1= [ (2*p*r)/(p+r) for p,r in zip(precision, recall)]
	
	taccuracy.append(accuracy)
	tprecision.append(precision)
	trecall.append(recall)
	tf1.append(f1)
	
	print "Run",run,"metrics:"
	print "TEST DATA"
	print "Accuracy=",accuracy
	print "Precision:"
	PrintMetricList(precision)
	print "Recall:"
	PrintMetricList(recall)
	print "F1:"
	PrintMetricList(f1)

	accuracy=Accuracy(confusionWhole, dataSizeInv)
	precision=Precision(confusionWhole)
	recall=Recall(confusionWhole)
	f1= [ (2*p*r)/(p+r) for p,r in zip(precision, recall)]
	
	daccuracy.append(accuracy)
	dprecision.append(precision)
	drecall.append(recall)
	df1.append(f1)
	
	print "Run",run,"metrics:"
	print "WHOLE DATA"
	print "Accuracy=",accuracy
	print "Precision:"
	PrintMetricList(precision)
	print "Recall:"
	PrintMetricList(recall)
	print "F1:"
	PrintMetricList(f1)


print "Final Statistics over test data:"
print "Accuracy: mean=", mean(taccuracy), "variance=", variance(taccuracy)

print "Precision mean:"
PrintMetricList(MeanOfEach(tprecision))
print "Precision variance:"
PrintMetricList(VarianceOfEach(tprecision))

print "Recall mean:"
PrintMetricList(MeanOfEach(trecall))
print "Recall variance:"
PrintMetricList(VarianceOfEach(trecall))

print "F1 mean:"
PrintMetricList(MeanOfEach(tf1))
print "F1 variance:"
PrintMetricList(VarianceOfEach(tf1))


print "Final Statistics over whole data:"
print "Accuracy: mean=", mean(daccuracy), "variance=", variance(daccuracy)

print "Precision mean:"
PrintMetricList(MeanOfEach(dprecision))
print "Precision variance:"
PrintMetricList(VarianceOfEach(dprecision))

print "Recall mean:"
PrintMetricList(MeanOfEach(drecall))
print "Recall variance:"
PrintMetricList(VarianceOfEach(drecall))

print "F1 mean:"
PrintMetricList(MeanOfEach(df1))
print "F1 variance:"
PrintMetricList(VarianceOfEach(df1))

