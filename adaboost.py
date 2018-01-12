import numpy as np
from sklearn import linear_model
import math
from copy import deepcopy
from sklearn import metrics
import sklearn
from sklearn import naive_bayes
import scipy.sparse as sparse


""" 
This model is based on Sklearn 

"""
import matplotlib.pyplot as plt


class AdaBoost():

    def __init__(self, debug=False):
        """
        """
        self.MODELS=[]
        self.ALPHAS=[]
        self.debug = debug

    """ This is the selected nase learner """
    def learner(self,X,y,W):
        model = linear_model.LogisticRegression(C=sum(W))
        model.fit(X,y,sample_weight=W)
        #adjust threshold according to the AUC curve
        #all of the pos probs
        #fpr, tpr, threshold= metrics.roc_curve(y,preds,sample_weight=W)
        return model


    def fit(self, train, label, M=30):
        """
        :param X: training data
        :param y: training label
        :return:
        """
        #initialize W
        X = deepcopy(train)
        y = deepcopy(label)
        X, y = np.asarray(X), np.asarray(y)
        #change label to +1 and -1
        y[y==0] = -1
        MODEL_LIST=[]
        ALPHA_LIST=[]
        n = X.shape[0]
        W = np.ones(n) / n

        for t in range(M):
            print("-----iteration {} -------".format(t+1))
            model = self.learner(X,y,W)
            preds =np.asarray(model.predict(X))
            miss = [int(x) for x in (preds != y)]
            error = np.dot(W, miss)
            if error ==0:
                alpha = 1/2* np.log((1 - 0.98) / float(0.98))
            if error != 0:
                alpha = 1/2* np.log((1 - error) / float(error))

            if error == 0.5:
                alpha = 1/2 * np.log((1 - 0.51) / float(0.51))
            if self.debug:
                print("10 weights before updates", W[-10:])
            Z = np.dot( W, np.exp(-1* alpha * np.multiply(preds,y)))
            W = np.multiply(W, np.exp(-alpha * np.multiply(preds,y)))/Z


            MODEL_LIST.append(model)
            ALPHA_LIST.append(alpha)

            if self.debug:
                print("current error is: ",error)
                print("current alpha is: ", alpha)
                print("first 10 weights",W[:10])
                print("first 10 preds",preds[:10])
                print("first 10 labels", y[:10])
                print("last 10 weights", W[-10:])
                print("model prediction: ", preds[-10:])
                print("labels:          ",y[-10:])
                print(miss[-10:])
                print("\n")

        self.MODELS, self.ALPHAS = MODEL_LIST, ALPHA_LIST


    def predict(self, test_set):
        """"""
        test = deepcopy(test_set)
        if self.MODELS == [] or self.ALPHAS == []:
            raise ValueError("fit a model before predict")
        n = test.shape[0]
        test = test
        preds = np.zeros(n)
        for i in range(len(self.MODELS)):
            model,alpha = self.MODELS[i], self.ALPHAS[i]
            p = np.asarray(model.predict(test))
            p = alpha * p
            preds = np.add(preds,p)
            if self.debug:
                print("\n------predict round {} ------".format(i+1))
                print("alpha is", alpha)
                print("round prediction(last 10): ", p[-10:])
                print("current preds, ", preds[-10:])
        preds = np.sign(preds)
        return preds



    def predict_and_calculate_error(self,test,y):
        """"""
        test = np.asarray(deepcopy(test))
        preds = self.predict(test)
        label = deepcopy(y)
        label = np.asarray(label)
        label[label==0] = -1
        c = np.sum([x for x in preds == label])
        c = c/len(preds)
        error = 1-c
        c *= 100
        error *= 100
        c=round(c,3)
        error=round(error,3)
        print("\n-----TEST INFORMATION-----: \nnumber of test sample :{} \ncorrect rate is :{}%  \nerror is: {}% \n----------------------".format(len(preds), c ,  error) )
        return preds





    def draw_roc_cur(self,true_labels, scores):
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, scores)
        roc_auc = metrics.auc(fpr, tpr)  # compute area under the curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        ax2 = plt.gca().twinx()
        ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
        ax2.set_ylabel('Threshold', color='r')
        ax2.set_ylim([thresholds[-1], thresholds[0]])
        ax2.set_xlim([fpr[0], fpr[-1]])
        plt.savefig('roc_and_threshold.png')
        plt.close()



    def random_sampling(self, X,y,W):
        """this creates a random sampling of the data

            W is the probability of drawing each sample
        """
        n= X.shape[0]
        a= np.arange(0,n)
        ind_list = np.random.choice(a=a,p=W,size=n)
        new_X, new_y = [], []
        for ind in ind_list:
            new_X.append(X[ind])
            new_y.append(y[ind])
        new_X = np.asarray(new_X)
        new_y = np.asarray(new_y)

        print("weight distribution", sum(new_y))

        #lr = linear_model.LogisticRegression()
        #lr.fit(new_X,new_y)
        #preds = lr.predict(new_X)
        #c = np.sum([x for x in preds == y])
        #c = c/len(preds)
        #print("my model error is ",c)

        return new_X,new_y

    def LR(self):
        """"""
