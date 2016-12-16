import numpy as np
import scipy.io
import math

class Cry:
    def _init_():
        cry = 0
    
    #Gets the .mat from a certain direction
    def createMatrix (self, typeCry):
        if typeCry == "Asphyxia":
            cry = scipy.io.loadmat('C:/Users/Liz/Desktop/Llantos/LlantoAsfixN.mat', squeeze_me=True, struct_as_record=False)
        if typeCry == "Normal":
            cry =  scipy.io.loadmat('C:/Users/Liz/Desktop/Llantos/LlantoNormLyap.mat', squeeze_me=True, struct_as_record=False)
        if typeCry == "Deaf":
            cry = scipy.io.loadmat('C:/Users/Liz/Desktop/Llantos/LlantoSordoLyap.mat', squeeze_me=True, struct_as_record=False)
        return cry
    
    #Sets the experts with the Gaussian function
    def setExperts(self, expert, standardExpert, meanCry, varianceExpert):
        newExpert = np.zeros([len(expert)])        
        j = 0

        divpi = 1/(standardExpert * math.sqrt(2 * math.pi))
        for x in expert:
            resta = (math.pow(-(x-meanCry), 2))/(2* varianceExpert)
            newExpert[j] = divpi * math.pow(math.e, resta)
            j = j+1
        return newExpert
        
    #Sets the Maximum Likelihood
    def setMaximumLikelihood(self, cryColumn, expert):
        temp = 0    
        l = 0
        newML = np.zeros([len(cryColumn)]) 
        
        for y in cryColumn:
            for h in expert:
                temp = temp + (y * h)
            newML[l] = temp
            l = l + 1
        return newML
    
    #Scale the values
    def setScale(self, minimumA, maximumB, cryML, maxAll, minAll):
        scaled = np.zeros([len(cryML)]) 
        t = 0
        a = minimumA
        b = maximumB 
        
        maximum = maxAll#np.max(cryML)
        minimum = minAll#np.min(cryML)

        for i in cryML:
            scaled[t] = (((b-a) * (i - minimum)) / (maximum-minimum)) + a
            t = t + 1
        return scaled
    
    #Create the TXT for the cries 
    def createTXT(self, filename, scaledCry):
        listS = scaledCry.tolist()                        
        file = open(filename + ".txt", 'a')
        for x in listS:
            file.write(str(x))
            file.write('\n')
        file.close()
        
cryB = Cry()

asphyxiaMat = cryB.createMatrix("Asphyxia")
normalMat =  cryB.createMatrix("Normal")
deafMat =  cryB.createMatrix("Deaf")

#Eliminates the title 
sigM = asphyxiaMat['PCMtxAsfix']
sigM2 = normalMat['MtxNormLyap']
sigM3 = deafMat['MtxSordoLyap']

 #Transpose of the matrix
sigM1 = np.column_stack(sigM)
sigM21 = np.column_stack(sigM2)
sigM31 = np.column_stack(sigM3)

 #Create the 10 column cry
asphyxia10 = np.concatenate((sigM1[0], sigM1[1], sigM1[2], sigM1[3],
                        sigM1[4], sigM1[5], sigM1[6], sigM1[7],
                        sigM1[8], sigM1[9]),axis=0)

normal10 = np.concatenate((sigM21[0], sigM21[1], sigM21[2], sigM21[3],
                         sigM21[4], sigM21[5], sigM21[6], sigM21[7],
                         sigM21[8], sigM21[9]), axis=0)#sigM2[0]
                         
deaf10 = np.concatenate((np.float64(sigM31[0]), np.float64(sigM31[1]), np.float64(sigM31[2]), np.float64(sigM31[3]),
                         np.float64(sigM31[4]), np.float64(sigM31[5]), np.float64(sigM31[6]), np.float64(sigM31[7]), 
                         np.float64(sigM31[8]), np.float64(sigM31[9])),axis=0)#sigM3[0]

#Column 11 = expert                  
asphyxia11 =sigM1[10]
normal11 = sigM21[10]
deaf11 = sigM31[10]

#Mean, covariance, variance and standard deviation
#for every cry
meanAsphyxia = np.mean(asphyxia10)
covarianceAsphyxia = np.cov(asphyxia10)
meanAsphyxia11 = np.mean(asphyxia11)
covarianceAsphyxia11 = np.cov(asphyxia11)
varianceAsphyxia11 = np.var(asphyxia11)
standardAsphyxia11 = np.std(asphyxia11)

meanNormal = np.mean(normal10)
covarianceNormal = np.cov(normal10)
meanNormal11 = np.mean(normal11)
covarianceNormal11 = np.cov(normal11)
varianceNormal11 = np.var(normal11)
standardNormal11 = np.std(normal11)

meanDeaf = np.mean(deaf10)
covarianceDeaf = np.cov(deaf10)
meanDeaf11 = np.mean(deaf11)
covarianceDeaf11 = np.cov(deaf11)
varianceDeaf11 = np.var(deaf11)
standardDeaf11 = np.std(deaf11)

 #Creation of empty vectors
expertAsphyxia = np.zeros([len(asphyxia11)])
expertNormal = np.zeros([len(normal11)])
expertDeaf = np.zeros([len(deaf11)])

asphyxia = np.zeros([len(asphyxia10)])
normal = np.zeros([len(normal10)])
deaf = np.zeros([len(deaf10)])

scaledAsphyxia = np.zeros([len(asphyxia10)])
scaledNormal = np.zeros([len(normal10)])
scaledDeaf = np.zeros([len(deaf10)])

 #Set the experts with the Gaussian function
expertAsphyxia = cryB.setExperts(asphyxia11, standardAsphyxia11, meanAsphyxia, varianceAsphyxia11)
expertNormal = cryB.setExperts(normal11, standardNormal11, meanNormal, varianceNormal11)
expertDeaf = cryB.setExperts(deaf11, standardDeaf11, meanDeaf, varianceDeaf11)

 #Set the 10 colum cry with Maximum Likelihood 
asphyxia = cryB.setMaximumLikelihood(asphyxia10, expertAsphyxia)
normal = cryB.setMaximumLikelihood(normal10, expertNormal)
deaf = cryB.setMaximumLikelihood(deaf10, expertDeaf)

 #Scale the values
maximumA = np.max(asphyxia)
minimumA = np.min(asphyxia)
maximumD = np.max(deaf)
minimumD = np.min(deaf)
maximumN = np.max(normal)
minimumN = np.min(normal)

maxMin = [maximumA, minimumA, maximumD,minimumD, maximumN, minimumN]
Minmax = [maximumA, maximumD, maximumN]

maxiAll = np.min(Minmax)
miniAll = np.min(maxMin)

scaledAsphyxia = cryB.setScale(miniAll,maxiAll,asphyxia, maximumA, minimumA)
scaledNormal = cryB.setScale(miniAll,maxiAll,normal, maximumN, minimumN)
scaledDeaf = cryB.setScale(miniAll,maxiAll,deaf, maximumD, minimumD)

 #Creation of TXT
cryB.createTXT("Asphyxia", scaledAsphyxia)
cryB.createTXT("Normal", scaledNormal)
cryB.createTXT("Deaf", scaledDeaf)
