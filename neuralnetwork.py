from utils import *

def initialize_weights(ohe_category,layer_parameter, biasterm=True):
    weight_matrix_list = []

    inputcategory, outputcategory = [],[]
    inputindex, outputindex = [],[]
    n = 0
    for i in ohe_category:
        if ohe_category[i] != 'class_numerical':
            inputcategory.append(i) # name of the input category
            inputindex.append(n) # index of the input category
        else:
            outputcategory.append(i) # name of the output category  
            outputindex.append(n) # index of the output category
        n += 1
    
    b = 1 if biasterm == True else 0
    
    updatedlayerparameterwbias = [len(inputcategory)+b] + list(np.array(layer_parameter)+b) + [len(outputcategory)] # [inputlayer, layerparameters, outputlayer]
    for i in range(len(updatedlayerparameterwbias)-1):
        layernow = updatedlayerparameterwbias[i]
        layernext = updatedlayerparameterwbias[i+1]-1 if i !=len(updatedlayerparameterwbias)-2 else updatedlayerparameterwbias[i+1] 
        # ^ for the last layer, the bias is not included, so don't need to minus 1 ^
        init_weight = np.random.rand(layernext,layernow) * 2 - 1 # initialize the weight with random number between -1 and 1
        weight_matrix_list.append(init_weight)
        
    return weight_matrix_list

def costfunction(expected_output, actual_output):
    j = -np.multiply(expected_output,np.log(actual_output)) - np.multiply((1 - expected_output),np.log(1 - actual_output))
    return np.sum(j)

def sumofweights(listofweights,bias=True): # computes the square of all weights of the network and sum them up
    sum = 0
    for weight in listofweights:
        if bias:
            w = weight.copy()
            w[:, 0] = 0
            sum += np.sum(np.square(w))
        else:
            sum += np.sum(np.square(weight))
    return sum

def blame(predict_output, expected_output, weights_list, a_list, biasterm=True): # This is to find out the delta function
    deltalist = []
    delta_layer_l = predict_output - expected_output
    deltalist.append(delta_layer_l)
    i = len(weights_list)-1
    current_delta = delta_layer_l

    while i > 0:
        delta_layer_now = np.multiply(np.multiply(np.dot(weights_list[i].T,current_delta),a_list[i]),(1-a_list[i]))
        if biasterm:
            delta_layer_now[0] = 1 # the first attribute is the bias
            current_delta = delta_layer_now[1:] # the first attribute is the bias
        else:
            current_delta = delta_layer_now
        deltalist.append(current_delta)
        i-=1
    deltalist.reverse()
    
    return deltalist

def gradientD(weights_list,deltalist,attributelist,biasterm=True):
    gradlist = []
    for i in range(len(weights_list)):
        attributenow = attributelist[i]
        deltanow = np.array([deltalist[i]]).T
        dotproduct = deltanow*attributenow
        # print('dotshape',dotproduct.shape)
        gradlist.append(dotproduct)
    return gradlist

# Forward propagation vectorized
def neural_network(normed_hotted_data,ohe_category,weights_list, minibatchk = 15, lambda_reg = 0.2, learning_rate = 0.01):
    biasterm=True
    normed_ohe_copy = normed_hotted_data.copy()
    if minibatchk > len(normed_hotted_data):
        minibatchk = len(normed_hotted_data)
    np.random.shuffle(normed_ohe_copy)
    # print('minibatchk',minibatchk)
    # print('shape of normed_ohe_copy',normed_ohe_copy.shape)
    splitted = np.array_split(normed_ohe_copy, minibatchk)
    
    inputcategory, outputcategory = [],[]
    inputindex, outputindex = [],[]
    n = 0
    for i in ohe_category:
        if ohe_category[i] != 'class_numerical':
            inputcategory.append(i) # name of the input category
            inputindex.append(n) # index of the input category
        else:
            outputcategory.append(i) # name of the output category
            outputindex.append(n) # index of the output category
        n += 1
    
    b = 1 if biasterm else 0
    
    for onebatch in splitted:
        onebatch = onebatch.T
        input_data = onebatch[inputindex].T
        output_data = onebatch[outputindex].T
        # input_data_mean = onebatch[inputindex].mean(axis=1)
        output_data = onebatch[outputindex].T

        # forward propagation
        instance_index = 0
        j = 0
        listofgradient = []
        for one_instance in input_data:
            current_layer_a = np.append(1,one_instance) if b == 1 else one_instance
            # input layer is the current layer
            current_layer_index = 0
            output_expect = output_data[instance_index]
            attributesnobias = [one_instance]
            attributeswbias = [current_layer_a]
            for theta in weights_list:
                z = np.dot(theta,current_layer_a)
                a = g(z)
                current_layer_a = np.append(1,a) if (b == 1) and (current_layer_index+1 != len(weights_list)) else a
                attributesnobias.append(a)
                attributeswbias.append(current_layer_a)
                current_layer_index += 1

            output_predict = current_layer_a # the last attribute is the output for this batch.
            instance_index += 1
            j += costfunction(output_expect,output_predict)

            # calculate delta blame (back propagation)
            listofdelta = blame(output_predict,output_expect,weights_list, attributeswbias)
            thisgradient = gradientD(weights_list,listofdelta,attributeswbias,biasterm)
            listofgradient.append(thisgradient)
        
        gradientP = [lambda_reg*t for t in weights_list]
        # first column in singleP in the np.array = 0
        for singleP in gradientP:
            singleP[:, 0] = 0
        
        grad_D_transpose = transposelistoflist(listofgradient)
        grad_D_sum = [np.sum(t,axis=0) for t in grad_D_transpose]
        gradients_batch = []
        for i in range(len(grad_D_sum)):
            gradients_batch.append((grad_D_sum[i] + gradientP[i])*(1/instance_index))
        
        j /= (instance_index+1)
        s = sumofweights(weights_list,bias=b)*lambda_reg/(2*(instance_index+1))
        allj = j+s # total cose with regularization

        # update weights
        for i in range(len(weights_list)):
            weights_list[i] -= learning_rate*gradients_batch[i]

    return weights_list, allj, j #j is j without regularization: only for testing

def train_neural_network(normed_ohetraining_data,ohe_category,layerparameter, minibatchk = 15, lambda_reg = 0.15, learning_rate = 0.01, epsilon_0 = 0.00001, softstop = 8000, printq = False):
    init_weight = initialize_weights(ohe_category,layerparameter)
    updated_weight, jsum, purej = neural_network(normed_ohetraining_data,ohe_category,init_weight, minibatchk, lambda_reg, learning_rate)
    epsilon = epsilon_0 + 20
    currentj = jsum
    smallestj = jsum
    count = 0
    jlist = []
    jlist.append(currentj)
    while ((epsilon > epsilon_0) or (count < softstop) or (currentj >= smallestj)) and (count < (softstop)):
        if printq:
            print('currentj',currentj)
            print('count',count)
        count += 1
        updated_weight, jsum, purej = neural_network(normed_ohetraining_data,ohe_category,updated_weight,minibatchk,lambda_reg,learning_rate)
        epsilon = jsum - currentj
        currentj = jsum
        jlist.append(currentj)
        if currentj < smallestj:
            smallestj = currentj
    
    return updated_weight, jlist

def predictoneinstance(inputdata,weightl): # inputdata here doesn't include the class and bias.
    current_layer_a = np.append(1,inputdata)
    current_layer_index = 0
    alist = []
    alist.append(current_layer_a)
    for theta in weightl:
        z = np.dot(theta,current_layer_a)
        a = g(z)
        current_layer_a = np.append(1,a) if (current_layer_index+1 != len(weightl)) else a
        alist.append(current_layer_a)
        current_layer_index += 1
        raw_output = a
    predict_output = current_layer_a

    if len(predict_output) <=1:
        predict_output[0] = 0 if predict_output[0] <= 0.5 else 1
    else:
        predict_output[np.where(predict_output==np.max(predict_output))] = 1
        predict_output[np.where(predict_output!=1)] = 0
    
    return predict_output, raw_output 

def predict_many_nn(testdatafull, ohecategory, weight):
    n = 0 
    inputindex, outputindex = [],[]
    for i in ohecategory:
        if ohecategory[i] != 'class_numerical':
            inputindex.append(n)
        else:
            outputindex.append(n)
        n += 1
    predictvsexpectlist = [] # list of list of predict and expect/actual

    for instance in testdatafull:
        datainput = instance[inputindex]
        expect_output = instance[outputindex]
        predict_output, raw_output = predictoneinstance(datainput,weight)
        # process the index of value 1 in np.array
        processdexpect = np.where(expect_output==1)[0][0]
        processdpredict = np.where(predict_output==1)[0][0]
        predictvsexpectlist.append([processdpredict,processdexpect])

    correct = 0
    for outputtup in predictvsexpectlist:
        if outputtup[0] == outputtup[1]:
            correct += 1
    accuracy = correct/len(predictvsexpectlist)

    return predictvsexpectlist, accuracy
    