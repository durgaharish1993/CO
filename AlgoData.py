import numpy as np
from collections import defaultdict
import math


class Data():
    def __init__(self, x, w=None, h=None, n=None, m=None, r=None, noise=None):
        # Here initializing if x, w, h are known to us
        self.X_given = x
        self.W_given = w
        self.H_given = h
        self.n_given = n
        self.m_given = m
        self.r_given = r
        self.N_given = noise


class DataGeneration():

    def __init__(self,m,r,n):
        self.m  = m
        self.r  = r
        self.n  = n
        print("This Class Generates Data")

    # This function generates synthetic Data X = WH+N.
    @staticmethod
    def synthetic_data_generation(m,r,n,standard_deviation):
        stddev = standard_deviation
        #(m, r, n) = (self.m, self.r, self.n)

        noise = np.matrix(np.random.normal(loc=0, scale=stddev, size=(m, n)))
        H_prime = np.matrix(np.random.uniform(low=0, high=1, size=(r, n - r)))
        H_iden = np.matrix(np.eye(r))
        H_matrix = np.concatenate([H_iden, H_prime], axis=1)
        W_matrix = np.matrix(np.random.uniform(low=0, high=5, size=(m, r)))

        X_matrix = (W_matrix * H_matrix) + noise
        # print(W_matrix)
        # print(H_matrix)
        #
        # print("#######")
        #
        # print(X_matrix)

        return X_matrix, W_matrix, H_matrix, noise, (m, r, n)




class BBCData():
    '''
    1. This class provides data for BBC pre-processed data, 
    2. This class generates Matrix X with human labelled class (r)
    '''

    def __init__(self,folderPath,type='normal',basename='bbc', inverseRep = True):

        self.info         = " This class provides data for BBC, Downloaded from http://mlg.ucd.ie/datasets/bbc.html"
        self.folderPath   = folderPath

        self.filePath     = folderPath + basename + '.mtx'
        self.termPath     = folderPath + basename + '.terms'
        self.type         = type
        self.tfCountStat  = defaultdict(lambda : defaultdict(int))
        self.idCountStat  = defaultdict(int)
        self.docWordCount = defaultdict(int)
        self.inverseRep   = inverseRep


    def genTFIDFData(self):

        with open(self.filePath, 'r') as fp:
            fp.readline()
            [n,m,total_words]=list(map(int,fp.readline().strip().split(' ')))
            for line in fp.readlines():
                temp_list = list(map(float,line.strip().split(' ')))
                self.tfCountStat[temp_list[1]][temp_list[0]] = temp_list[2]
                self.idCountStat[temp_list[0]] +=1
                self.docWordCount[temp_list[1]] +=temp_list[2]


        complete_X = []
        for doc in range(1,m+1):
            temp_list = [0.0]* n
            for word in self.tfCountStat[doc]:
                tf_val                 = float(self.tfCountStat[doc][word])/ self.docWordCount[doc]
                id_val                 = math.log(float(m)/self.idCountStat[word])
                tfID_val               = tf_val * id_val
                temp_list[int(word-1)] = tfID_val


            complete_X+=[temp_list]

        if self.inverseRep:
            self.X = np.matrix(complete_X).T
            self.m = n
            self.n = m


        else:

            self.X = np.matrix(complete_X)
            self.m = m
            self.n = n


        return self.X,(self.m,self.n)




    def getWords(self,aIndex):
        words = []
        with open(self.termPath,'r') as fp:
            for line in fp.readlines():
                words +=[line.strip()]



        words       = np.array(words)
        anchorWords = words[aIndex]

        return anchorWords





    def getWordVectors(self,aIndex,topK=10):
        words = []
        with open(self.termPath,'r') as fp:
            for line in fp.readlines():
                words +=[line.strip()]



        words        = np.array(words)

        wordVectors  = self.X[:,aIndex]
        (m,r)        = wordVectors.shape
        score_matrix = []
        for i in range(r):
            cur_wordVector = wordVectors[:,i]
            word_indexes   = np.where(cur_wordVector > 0)[0]
            score_words    = np.matrix(cur_wordVector[cur_wordVector>0]).T
            cur_i_words    = words[word_indexes]
            np_cur_i_words = np.matrix(cur_i_words).T
            final_matrix   = np.concatenate([np_cur_i_words, score_words], axis=1)
            final_matrix   = final_matrix[np.argsort(final_matrix.A[:,1])] [:: -1]
            score_matrix   += [final_matrix[:topK,:]]


        return score_matrix












class PreprocessData():
    '''
    1. This class provides data for BBC pre-processed data, 
    2. This class generates Matrix X with human labelled class (r)
    '''

    def __init__(self,folderPath,type='normal',basename='bbc', inverseRep = True):

        self.info         = " This class provides data for BBC, Downloaded from http://mlg.ucd.ie/datasets/bbc.html"
        self.folderPath   = folderPath

        self.filePath     = folderPath + basename + '.mtx'
        self.termPath     = folderPath + basename + '.terms'
        self.type         = type
        self.tfCountStat  = defaultdict(lambda : defaultdict(int))
        self.idCountStat  = defaultdict(int)
        self.docWordCount = defaultdict(int)
        self.inverseRep   = inverseRep
        self.docum_term   = defaultdict(lambda : defaultdict(float))

    def genTFIDFData(self):

        with open(self.filePath, 'r') as fp:
            fp.readline()
            fp.readline()
            #n = words
            #m = documents
            [m,n,total_words]=list(map(int,fp.readline().strip().split(' ')))
            for line in fp.readlines():
                temp_list = list(map(float,line.strip().split(' ')))
                self.docum_term[temp_list[0]][temp_list[1]] = temp_list[2]
                # self.tfCountStat[temp_list[1]][temp_list[0]] = temp_list[2]
                # self.idCountStat[temp_list[0]] +=1
                # self.docWordCount[temp_list[1]] +=temp_list[2]


        complete_X = []
        for doc in range(1,m+1):
            temp_list = [0.0]* n
            for word in self.docum_term[doc]:

                # tf_val                 = float(self.tfCountStat[doc][word])/ self.docWordCount[doc]
                # id_val                 = math.log(float(m)/self.idCountStat[word])
                # tfID_val               = tf_val * id_val
                temp_list[int(word-1)] = float(self.docum_term[doc][word])


            complete_X+=[temp_list]

        if self.inverseRep:
            self.X = np.matrix(complete_X).T
            self.m = n
            self.n = m


        else:

            self.X = np.matrix(complete_X)
            self.m = m
            self.n = n


        return self.X,(self.m,self.n)




    def getWords(self,aIndex):
        words = []
        with open(self.termPath,'r') as fp:
            for line in fp.readlines():
                words +=[line.strip()]



        words       = np.array(words)
        anchorWords = words[aIndex]

        return anchorWords





    def getWordVectors(self,aIndex,topK=10):
        words = []
        with open(self.termPath,'r') as fp:
            for line in fp.readlines():
                words +=[line.strip()]



        words        = np.array(words)

        wordVectors  = self.X[:,aIndex]
        (m,r)        = wordVectors.shape
        score_matrix = []
        for i in range(r):
            cur_wordVector = wordVectors[:,i]
            if cur_wordVector.shape[0]<topK:
                print(cur_wordVector)
                continue

            word_indexes   = np.where(cur_wordVector > 0)[0]
            score_words    = np.matrix(cur_wordVector[cur_wordVector>0]).T
            cur_i_words    = words[word_indexes]

            np_cur_i_words = np.matrix(cur_i_words).T
            final_matrix   = np.concatenate([np_cur_i_words, score_words], axis=1)
            final_matrix   = final_matrix[np.argsort(final_matrix.A[:,1])] [:: -1]
            score_matrix   += [final_matrix[:topK,:]]


        return score_matrix

