from stratified import *
from parkinsons import *

raw_category_p, raw_data_p = dataforharry()
hiddenlayerp_1 = [4]
hiddenlayerp_2 = [4,4]
epochp_1 = 1000

loflofoutputsp_1, accp_1, lofjlistp_1 = kfoldcrossvalidneuralnetwork(raw_data_p,raw_category_p,hiddenlayerp_1,k=10,minibatchk=5,lambda_reg=0.1, learning_rate=0.1, epsilon_0=0.00001, softstop=epochp_1, printq=False)