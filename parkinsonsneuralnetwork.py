from stratified import *
from parkinsons import *

raw_category_p, raw_data_p = dataforharry()
hiddenlayerp_1 = [4]
hiddenlayerp_2 = [4,4]
epochp_1 = 1000

loflofoutputsp_1, accp_1, lofjlistp_1 = kfoldcrossvalidneuralnetwork(raw_data_p,raw_category_p,hiddenlayerp_1,k=4,minibatchk=5,lambda_reg=0.1, learning_rate=0.1, epsilon_0=0.00001, softstop=epochp_1, printq=False)

accuarcyp_1, precisionp_1, recallp_1, fscore_p_1= meanevaluation(loflofoutputsp_1,1)
print("First House Data Neural Network with " + str(hiddenlayerp_1) + " hidden layers and " + str(epochp_1) + " epochs")
print("Accuracy:", accp_1)
print("F-score:", fscore_p_1)
accprint = 'Accuracy is ' + str( float("{0:.3f}". format(accp_1)))
fscprint = 'F-Score is ' + str( float("{0:.3f}". format(fscore_p_1)))
plt.plot(range(epochp_1+1), lofjlistp_1[1])
plt.xlabel("epoch")
plt.ylabel("J")
plt.title("First House Data Neural Network with " + str(hiddenlayerp_1) + " hidden layers and " + str(epochp_1) + " epochs")
# plt.axis([0,10,0,10])
# plt.text(0.95,0.01,accprint,fontsize=10)
plt.show()