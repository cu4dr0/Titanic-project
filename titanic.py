import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
trDf = pd.read_csv("train.csv")
teDf = pd.read_csv("test.csv")
genDf = pd.read_csv("gender_submission.csv")
#print("\t\t\t\tTRAIN\n")
#print("\t\tHEAD\n")
#print(trDf.head())
#print("\t\tINFO\n")
print(trDf.info())
#print("\t\tDESCRIBE\n")
#print(trDf.describe())
#print(trDf.isnull().sum())
trDf.drop("Cabin",axis = 1, inplace = True)
##cabin having too many missing values
"""
filling empty values
"""
#print(trDf["Age"].describe())
##mean 29.699118
#sns.countplot(x = "Survived", hue = "Pclass", data = trDf)
#plt.show()
#sns.barplot(x = "Pclass", y = "Age", data = trDf)
#plt.show()
#print(trDf[trDf["Pclass"] == 1 ].describe())
#print(trDf[trDf["Pclass"] == 2 ].describe())
#print(trDf[trDf["Pclass"] == 3 ].describe())
##Mean per class
#class 1: 38.233441
#class 2 : 29.877630
#class 3 : 25.140620
def meanAge(x):
	age = x[0]
	pClass = x[1]
	if pd.isnull(age):
		if pClass == 1:
			return 38
		elif pClass == 2:
			return 30
		else:
			return 25
	else:
		return x
trDf["Age"] = trDf[["Age", "Pclass"]].apply(meanAge, axis = 1)
#print(trDf.isnull().sum()["Age"])
#print(trDf["Embarked"].describe())
#print(trDf["Embarked"].head())
##only 2 missing of embarked, let´s drop them both
"""
cleaning letters values
"""
#print(trDf["Name"].unique())
#print(trDf["Name"].nunique())
##889
#print(trDf["Sex"].unique())
#print(trDf["Sex"].nunique())
##2
#print(trDf["Ticket"].unique())
#print(trDf["Ticket"].nunique())
##680
#print(trDf["Embarked"].unique())
#print(trDf["Embarked"].nunique())
##3
def sets(x):
	dic = {'male':1, 'female':0}
	return dic[x]
def embark(x):
	dic = {'S':0, 'C':1, 'Q':2}
	return dic[x]
trDf["Sex"] = trDf["Sex"].apply(sets)
#print(trDf.describe()["Sex"])
trDf.dropna(inplace = True)
trDf["Embarked"] = trDf["Embarked"].apply(embark)
trDf2 = trDf.copy()
#print(trDf.isnull().sum())
#print(trDf.describe()["Embarked"])
##cleaned data
##filled correctly 
#print("\t\t\t\tTest\n")
#print("\t\tHEAD\n")
#print(teDf.head())
#print("\t\tINFO\n")
#print(teDf.info())
#print("\t\tDESCRIBE\n")
#print(teDf.describe())
#print(teDf.isnull().sum())
##Cabin again, too many lost values, drop it
teDf.drop("Cabin", axis = 1, inplace = True)
"""
filling missing data
"""
teDf["Age"] = teDf[["Age", "Pclass"]].apply(meanAge, axis = 1)
#print(teDf.isnull().sum()["Age"])
##cleaned
#teDf.dropna(inplace = True)
teDf["Embarked"] = teDf["Embarked"].apply(embark)
teDf["Sex"] = teDf["Sex"].apply(sets)
#print(teDf.isnull().sum())
#print(teDf["Fare"].unique())
#print(teDf.mean()["Fare"])
##35.6271884892086
def fare(x):
	if pd.isnull(x):
		return 35.6271884892086
	else:
		return x
teDf["Fare"] = teDf["Fare"].apply(fare)
##everything cleaned
#print("\t\t\t\tGENDER\n")
#print("\t\tHEAD\n")
#print(genDf.head())
#print("\t\tINFO\n")
#print(genDf.info())
#print("\t\tDESCRIBE\n")
#print(genDf.describe())
##Gender is for knowing how data has to be sent to the platform
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
def prepareX(xTr, xTe):
	oe = OrdinalEncoder().fit(xTr)
	xTr = oe.transform(xTr)
	xTe = oe.transform(xTe)
	return xTr,xTe
def prepareY(yTr, yTe):
	le = LabelEncoder().fit(yTr)
	yTr = le.transform(yTr)
	yTe = le.transform(yTe)
	return yTr,yTe
def dealText(series,colTextName):
	import string
	"""1.1 Number of Words"""
	series["wordCount"+colTextName] = series[colTextName].apply(lambda x: len(str(x).split(" ")))
	"""1.2 Number of characters"""
	series["charCount"+colTextName] = series[colTextName].apply(len)
	"""1.3 Average Word Length"""
	def avgWord(sentence):
		words = sentence.split(" ")
		return (sum(len(word) for word in words)/len(words))
	series["avgWord"+colTextName] = series[colTextName].apply(lambda x: avgWord(x))
	"""1.4 Number of stopwords"""
	from nltk.corpus import stopwords
	stop = stopwords.words("english")
	series['stopwords'+colTextName] = series[colTextName].apply(lambda x: len([x2 for x2 in x.split() if x in stop]))
	"""1.5 Number of special characters"""
	series["special"+colTextName] = series[colTextName].apply(lambda x: len([x2 for x2 in x if x2 in string.punctuation]))
	"""1.6 Number of numerics"""
	series["numerics"+colTextName] = series[colTextName].apply(lambda x: len([x2 for x2 in x.split() if x2.isdigit()]))
	"""1.7 Number of Uppercase words"""
	series["upper"+colTextName] = series[colTextName].apply(lambda x: len([x2 for x2 in x.split() if x2.isupper()]))
	"""1.8 Encoders"""
	
	"""3.4 Term Frequency – Inverse Document Frequency (TF-IDF)"""
	#from sklearn.feature_extraction.text import TfidfVectorizer
	#tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
	#train_vect = tfidf.fit_transform(series[colTextName])
	#print(train_vect.shape)
	"""2.1 Lower case"""
	#series[colTextName] = series[colTextName].apply(lambda x : "".join(x2.lower() for x2 in x.split()))
	"""2.2 Removing Punctuation AND 2.3 Removal of Stop Words"""
	#def textProcessor(mess):
	#	"""
	#	1- remove punc
	#	2- remove stopwords
	#	3- return list of clean text words
	#	"""
	#	nopunc = [char for char in mess if char not in string.punctuation]
	#	nopunc = "".join(nopunc)
	#	nopunc = [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]
	#	return str(nopunc)[2:-2]
	#series[colTextName] = series[colTextName].apply(lambda x: textProcessor(x))
	"""2.4 10 most common word removal"""
	#desiredNumber = 10
	#freq = pd.Series(" ".join(series[colTextName]).split()).value_counts()[:desiredNumber]
	#series[colTextName] = series[colTextName].apply(lambda x: " ".join(x2 for x2 in x.split() if x2 not in freq))
	"""2.5 Rare words removal"""
	#freq = pd.Series(" ".join(series[colTextName]).split()).value_counts()[-desiredNumber:]
	#series[colTextName] = series[colTextName].apply(lambda x: " ".join(x2 for x2 in x.split() if x2 not in freq))
	"""2.6 Spelling correction"""
	"""
	from textblob import TextBlob #conda install -c conda-forge textblob OR pip install -U textblob
	series[colTextName] = series[colTextName][:].apply(lambda x: str(TextBlob(x).correct()))
	"""
	##takes a lot of time, un comment if you need it
	return series
def searchType(x):
	x = str(x)
	ind = x.find(".")
	ind2 = x.find(",")
	return x[ind2+2: ind]
##Searchs for words such as "madame", "sir" and more
trDf["Rank"] = trDf["Name"].apply(lambda x: searchType(x))
teDf["Rank"] = teDf["Name"].apply(lambda x: searchType(x))
def searchCountry(x):
	dic = {'Sir':"ENG", 'Mlle':"FR", 'Col':"ENG", 'Capt':"ENG", 'the Countess':"ENG", 'Jonkheer':"NETH",\
	'Mr':"ENG", 'Mrs':"ENG", 'Miss':"ENG", 'Master':"ENG", 'Don':"ESP", 'Rev':"ENG", 'Dr':"ENG", 'Mme':"FR",\
	'Ms':"ENG", 'Major':"ENG", 'Lady':"ENG","Dona": "ESP"}
	return dic[x]
##Because we have abreviation suffix such as "mme" (in french) we categorize them by their country
trDf["Country"] = trDf["Rank"].apply(lambda x: searchCountry(x))
teDf["Country"] = teDf["Rank"].apply(lambda x: searchCountry(x))
def categoryCountry(x):
	dic = {"ENG":0, "FR":1, "ESP":2, "NETH":3}
	return dic[x]
##we transform the above method into category numbers
trDf["Country"] = trDf["Country"].apply(lambda x: categoryCountry(x))
teDf["Country"] = teDf["Country"].apply(lambda x: categoryCountry(x))
def rank(x):
	dic = {'Mr':1, 'Mrs':1,'Miss':1,'Ms':1, 'Sir':2, 'Mlle':1, 'Col':3, 'Capt':3, 'the Countess':1, 'Jonkheer':0,\
	  'Master':3, 'Don':1, 'Rev':2, 'Dr':3, 'Mme':2, 'Major':3, 'Lady':1,\
	"Dona": 1}
	##mrs, miss and ms are the same, just as the male version mr
	##don and dona are the same as in spanish
	##mlle = mademoiselle, mme = madame
	##jhonkheer = lowest class in netherlands and belgica
	return dic[x]
##we give values according to their "ranks" (madame, sir, etc.) if they mean the same but it´s on another language
#the value still
trDf["Rank"] = trDf["Rank"].apply(lambda x: rank(x))
teDf["Rank"] = teDf["Rank"].apply(lambda x: rank(x))
def ticketRank(x):
	ind = 999
	try:
		x = int(x)
		return "X"
	except:
		ind = x.find(" ")
		return x[:ind]
##tickets have a unique value e.g('A/5 21171' have a "A/5" on it, and many more)
trDf["ticketRank"] = trDf["Ticket"].apply(lambda x: ticketRank(x))
teDf["ticketRank"] = teDf["Ticket"].apply(lambda x: ticketRank(x))
def noperiods(x):
	x = str(x).split(".")
	x = str(x)
	ind = x.find("'")
	ind2 = x.find("'", ind+1)
	return x[ind+1:ind2]
##some of the tickets have the same value but sepparated by periods, lets take them all out
trDf["ticketRank"] = trDf["ticketRank"].apply(lambda x: noperiods(x))
teDf["ticketRank"] = teDf["ticketRank"].apply(lambda x: noperiods(x))
def categoricalTicket(x):
	dic = {'A/5':1, 'PC':2, 'STON/O2':3, 'X':4, 'PP':5, 'C':6, 'A':7, 'SC/Paris':8, 'S':9, 'A/4':10, 'CA':11, 'SO/C':12,\
	'W':13, 'SOTON/OQ':14, 'STON/O':15, 'A4':16, 'SOTON/O':17, 'SC/PARIS':18, 'Fa':19, 'LIN':20, 'F':21, 'W/C':22,\
	'SW/PP':23, 'SCO/W':24, 'P/PP':25, 'SC':26, 'SC/AH':27, 'A/S':28, 'WE/P':29, 'SOTON/O2':30,'SC/A':31,\
	'STON/OQ':32,'SC/A4':33,'AQ/4':34, 'LP':35, 'AQ/3':36}
	##ston = 2
	##a = 1
	##W = 3
	##sw = 4
	#SCO = 5
	##SC = 6
	##AQ = 7
	##WE = 8
	##P = 9
	##SOTON = 10
	##SO = 11
	##CA = 12
	return dic[x]
##Once they all gone then proceed to give them values
trDf["ticketRank"] = trDf["ticketRank"].apply(lambda x: categoricalTicket(x))
teDf["ticketRank"] = teDf["ticketRank"].apply(lambda x: categoricalTicket(x))
def parenthesis(x):
	x = str(x)
	try:
		ind = x.find("(")
		ind2 = x.find(")")
		return x[ind:ind2+1]
	except:
		pass
trDf["parenthesis"] = trDf["Name"].apply(lambda x: parenthesis(x))
trDf = dealText(trDf,"parenthesis")
teDf["parenthesis"] = teDf["Name"].apply(lambda x: parenthesis(x))
teDf = dealText(teDf,"parenthesis")
def numberOfWords(x):
	x = str(x)
	return len([x2 for x2 in x.split(" ")])-1
trDf["parenthesis"] = trDf["parenthesis"].apply(lambda x: numberOfWords(x))
teDf["parenthesis"] = teDf["parenthesis"].apply(lambda x: numberOfWords(x))
def splitter1(x):
	x = str(x)
	ind = x.find(".")
	short = x[:ind]
	return short
trDf["short"] = trDf["Name"].apply(lambda x: splitter1(x))
trDf = dealText(trDf,"short")
teDf["short"] = teDf["Name"].apply(lambda x: splitter1(x))
teDf = dealText(teDf,"short")
def splitter2(x):
	ind = x.find(".")
	ind2 = x.find("(")
	name = x[ind+1:ind2]
	return name
trDf["name"] = trDf["Name"].apply(lambda x: splitter2(x))
trDf = dealText(trDf,"name")
teDf["name"] = teDf["Name"].apply(lambda x: splitter2(x))
teDf = dealText(teDf,"name")
trDf["name"] = trDf["name"].apply(lambda x: numberOfWords(x))
teDf["name"] = teDf["name"].apply(lambda x: numberOfWords(x))
trDf["short"] = trDf["short"].apply(lambda x: numberOfWords(x))
teDf["short"] = teDf["short"].apply(lambda x: numberOfWords(x))

print("CHECK HERE MATE")
print(trDf.info())
print(trDf["short"].head())
print(trDf["name"].head())

trDf.drop("Name",axis =1, inplace = True)
trDf.drop("Ticket",axis = 1, inplace = True)
teDf.drop("Name", axis = 1, inplace = True)
teDf.drop("Ticket", axis = 1, inplace = True)
X = trDf.drop("Survived", axis = 1).values
y = trDf["Survived"].values
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer,\
PowerTransformer
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
xTrain,xTest, yTrain,yTest = train_test_split(X,y,test_size = 0.25)
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)
from keras.callbacks import EarlyStopping
earlyStop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 60, verbose = 1)
from keras.models import Sequential
from keras.layers import Dropout, Dense
model = Sequential()
print(xTrain.shape)
##15
model.add(Dense(15,activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(8,activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(4,activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(2,activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(1, activation  = "sigmoid"))
model.compile(loss = "binary_crossentropy", optimizer = "adam")
model.fit(xTrain,yTrain, epochs = 9999, validation_data = (xTest, yTest), callbacks = [earlyStop])
#losses = pd.DataFrame(model.history.history)
#losses.plot()
predictions = model.predict_classes(xTest)
from sklearn.metrics import confusion_matrix, classification_report
print(xTrain.shape)
print(confusion_matrix(yTest,predictions))
print(classification_report(yTest, predictions))
print(type(predictions))
#print(teDf.info())
#print(teDf.head())
#print(model.predict_classes(teDf))
predict = scaler.transform(teDf)
predict = model.predict_classes(predict)
def numpyNoBrackets(theArray):
	result = []
	for eachNumber in range(len(theArray)):
		result.append(theArray[eachNumber][0])
	return result
predictions = numpyNoBrackets(predictions)
predict = numpyNoBrackets(predict)
pred = np.array(predict)
print(pred)
##from tensor to numpy so I get no floats
passID = np.array(teDf["PassengerId"])
##from tensor to numpy so I get no floats
#print(pred)
#print(passID)
df = pd.DataFrame(pred)
##from numpy to pandas
df2 = pd.DataFrame(passID)
##from numpy to pandas
saveMe = pd.concat([df2,df], axis = 1)
saveMe.columns = "PassengerId", "Survived"
print(saveMe)
try:
	saveMe.to_csv("survivors.csv",index=False)
	pass
except:
	pass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
pipe = Pipeline([
	("bow",CountVectorizer()),
	("tfid",TfidfTransformer()),
	("classifier",MultinomialNB())
	])	
xTrain2,xTest2,yTrain2,yTest2 = train_test_split(trDf2["Name"], trDf["Survived"], test_size = 0.3, random_state = 101)
pipe.fit(xTrain2,yTrain2)
predictions2 = pipe.predict(xTest2)
print(confusion_matrix(yTest2,predictions2))
print(classification_report(yTest2, predictions2))
predict2 = pipe.predict(teDf)
print(predict2)
plt.show()