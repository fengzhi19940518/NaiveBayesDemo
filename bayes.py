from numpy import *
import  re
import feedparser
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])

    for ducument in dataSet:
        vocabSet = vocabSet | set(ducument)
    #print("vocab",list(vocabSet))
    return list(vocabSet)


def bagOfWord2Vec(vocabList, inputSet):     #词袋模型
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCategory):    #训练函数
    numTrainDocs= len (trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = zeros(numWords)
    # p1Num=zeros(numWords)
    # p0Denom = 0.0
    # p1denom = 0.0
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1denom +=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num / p1denom
    # p0Vect = p0Num / p0Denom
    # 将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = log(p1Num / p1denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):            #分类函数
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():        #测试函数
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    print("myVocabList:",myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bagOfWord2Vec(myVocabList,postinDoc))
    print("bagOfWord2Vec:",bagOfWord2Vec(myVocabList, listOPosts[3]))

    p0V,p1V,pAb = trainNB0(trainMat , listClasses)

    testEntry = ['love','my','dalmation']
    thisDoc = array(bagOfWord2Vec(myVocabList,testEntry))

    print(testEntry,"classified as:",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']

    thisDoc = array(bagOfWord2Vec(myVocabList, testEntry))
    print(testEntry, "classified as:", classifyNB(thisDoc, p0V, p1V, pAb))

    # print(setofWord2Vec(myVocabList,listOPosts[0]))
    print("pAb:",pAb)
    print("p0V：",p0V)
    print("p1V：",p1V)
# testingNB()

#使用朴素贝叶斯进行交叉验证,文件解析及完整的垃圾邮件测试函数
def textParse(bigStirng):
    """
    文本切分
    输入文本字符串，输出词表
    """
    listOfTokens = re.split(r'\W*',bigStirng)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    '''
    #朴素贝叶斯的应用，垃圾邮件的过滤.垃圾邮件测试函数
    '''
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        #读取垃圾邮件
        wordList = textParse(open('email/spam/%d.txt' % i,'r',encoding= 'utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        #设置垃圾邮件类标签为1
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,'r',encoding= 'utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#生成次表库
    trainingSet = list(range(50))
    testSet=[]           #
    #随机选10组做测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []  #生成训练矩阵及标签
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #测试并计算错误率
    for docIndex in testSet:    #对测试集进行分类
        wordVector = bagOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText
# spamTest()


# mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
# print(mySent.split())
# #
# regEx = re.compile('\\W*')      #除单词，数字外的任意字符串
# listOfTokens = regEx.split(mySent)
# print(listOfTokens.__sizeof__())
# print([tok.lower() for tok in listOfTokens if len(tok)>0])
# emailText = open('./email/ham/6.txt').read()
# listOfTokens = regEx.split(emailText)
# print(listOfTokens.__sizeof__())

def calcMostFreq(vocabList, fullText):
    """
    :param vocabList: 词表
    :param fullText:
    :return: 返回前面三十个高频词
    """
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)

    return sortedFreq[:30]


'''
函数localWords()与程序清单中的spamTest()函数几乎相同，区别在于这里访问的是
RSS源而不是文件。然后调用函数calcMostFreq()来获得排序最高的30个单词并随后将它们移除
'''

def localWords(feed1, feed0):
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen));
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
    # testingNB()
   spamTest()
    # 导入RSS数据源
   # import operator
   # ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
   # sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
   # localWords(ny,sf)
   # getTopWords(ny,sf)
