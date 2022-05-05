from numpy import save
from stratified import *
from titanic import *

titan_rawcategory, titan_rawdata = titanic_dataforharry()
hidden1 = [4]
hidden2 = [8]
hidden3 = [8,8]
hidden4 = [8,8,8]
listoflayers = [hidden1,hidden2,hidden3,hidden4]
epochp_1 = 4000

def savef(rawdata,rawcategory,hiddenlayer,epoch,filename):
    loflofoutputs, acc, lofj = kfoldcrossvalidneuralnetwork(rawdata,rawcategory,hiddenlayer,k=10,minibatchk=8,lambda_reg=0.1, learning_rate=0.01, epsilon_0=0.00001, softstop=epoch, printq=False)
    accuracyp, precisionp, recallp, fscore_p= meanevaluation(loflofoutputs,1)
    plt.figure()
    print("Titanic Data Neural Network with " + str(hiddenlayer) + " hidden layers and " + str(epoch) + " epochs")
    print("Accuracy:",  float("{0:.4f}". format(acc)))
    print("F-score:",  float("{0:.4f}". format(fscore_p)))
    plt.plot(range(epoch+1), lofj[1])
    plt.xlabel("epoch")
    plt.ylabel("J")
    plt.title("Titanic Data Neural Network with " + str(hiddenlayer) + " hidden layers and " + str(epoch) + " epochs")
    plt.savefig("nnfig/titanic_nn_{}.png".format(filename))


filenames = ['4','8','88','888']

n = 0
for layer in listoflayers:
    savef(titan_rawdata,titan_rawcategory,layer,epochp_1+1000*n,filenames[n])
    n+=1

# Titanic Data Neural Network with [4] hidden layers and 4000 epochs
# Accuracy: 0.8208
# F-score: 0.8597

# Titanic Data Neural Network with [8] hidden layers and 5000 epochs
# Accuracy: 0.8186
# F-score: 0.8581

# Titanic Data Neural Network with [8, 8] hidden layers and 6000 epochs
# Accuracy: 0.8321
# F-score: 0.8676