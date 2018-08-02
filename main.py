'''
This project tries to replicate the Conic Hull Non Negative Matrix Factorization Paper. 
'''
from cvxpy import *
import numpy as np
from Algo import Xray
from AlgoData import Data,DataGeneration,BBCData,PreprocessData
#np.random.seed(1)
import datetime
import csv
import sys
import time





if __name__ =='__main__':



    '''
    Choice 1    : This is for synthetic Dataset 
    Choice 2    : This is for  real world Dataset, bbc, bbcsport, guardian, 
    base_name   : This is for selecting the real data set.
    '''

    choice = 2

    '''
    Once you select the choice, 
    if your choice is choice =1 or 2  
    then parameters to be chosen, 
    m = number of words 
    n = number of documents, 
    r = inner dimension of the matrix. 
    
    
    This is not completely modularized,  Please go inside the code block and change the parameters. 
    
    If any problem with running the code, pls contact dayapuld@oregonstate.edu. 
    
    
    
    '''


    if choice ==1:

        m, r, n = (210, 10, 200)
        output_list = [] #[['m','r','n','StdDev','Type','Not_Recovered','Mis_Match_Rate','loss','Run_Time']]
        filename    = 'Result_' + str(m)+'_'+str(r) + '_' + str(n) +"_Stddev"
        for i in range(1):
            for std in  [0]: #[0,0.2,0.4,0.5,0.6,0.8,1,1.2,1.4]: #0.2,0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5]:
                print('Current Standard Deviation: ',std)
                X, W, H, N, (m, r, n)=DataGeneration.synthetic_data_generation(m,r,n,standard_deviation=std)
                #X,W,H,N,(m,r,n)=d.synthetic_data_generation()


                raw_data ={

                    "x_given"       : X,
                    "w_given"       : W,
                    "h_given"       : H,
                    "n_given"       : n,
                    "m_given"       : m,
                    "r_given"       : r,
                    "ext_selection" : 'max',
                    'r_expected'    : r,
                    'random_seed'   : False,
                    'processes'     : 15,
                    'parallel_mode'  : False

                }

                s_time   = datetime.datetime.now()
                algo_ins = Xray(all_data=raw_data)

                aIndexes    = algo_ins.run_algoritm_synthetic()
                loss        = algo_ins.getLossValue()

                e_time   = datetime.datetime.now()

                run_time = (e_time-s_time).total_seconds()

                given_index   = range(r)
                out_index     = set(aIndexes) ^ set(given_index)
                out_index     = [i for i in out_index if i <= r]
                mismatch_rate = 1- (len(out_index)/float(len(given_index)))



                output_list+=[[m,r,n,std,raw_data['ext_selection'],len(out_index),mismatch_rate,loss, run_time]]

        with open('./results/Synthetic_data_result_Dist_R_30.csv','a') as fp:
            writer = csv.writer(fp)
            writer.writerows(output_list)







    if choice ==2:
        #, filePath = './raw_data/bbcsport/bbcsport.mtx'
        # bbc_data = BBCData(folderPath='./raw_data/bbc/',basename='bbc',type='normal',inverseRep = True)
        # X,(m,n)  = bbc_data.genTFIDFData()

        #Dataset 1 : guardian2013         (m , n , r )  =
        #Dataset 2 : irishtimes2013    [55, 581, 846, 1403, 1835, 1892, 1956]

        output_folder  = './results/'
        base_name      = 'irishtimes2013'
        k              = 10
        r              = 7

        inverse_rep    = True
        bbc_data  = PreprocessData(folderPath='./raw_data/preprocessed_data/'+base_name+'/', basename=base_name, type='normal', inverseRep=inverse_rep)
        X, (m, n) = bbc_data.genTFIDFData()



        raw_data ={

            "x_given"       : X,
            "w_given"       : None,
            "h_given"       : None,
            "n_given"       : n,
            "m_given"       : m,
            "r_given"       : r,
            "ext_selection" : 'dist',
            'r_expected'    : r,
            'random_seed'   : False,
            'processes'     : 15,
            'parallel_mode' : True

        }
        output_file = output_folder + 'Output_Document_' + base_name + '_K_' + str(k) + '_Type_'+raw_data['ext_selection']+'.csv'

        start_time = datetime.datetime.now()

        algo_ins   = Xray(all_data=raw_data)

        aIndexes   = algo_ins.run_algoritm_synthetic() #[2376, 2722, 3189, 4726, 4862, 5173]
        end_time   = datetime.datetime.now()

        timetaken  =  (end_time-start_time).total_seconds() # 965.617187 #(
        print('Time taken to run the algorithm:',timetaken)
        #words      = bbc_data.getWords(aIndexes)
        #print(words)
        if inverse_rep:
            word_vectors =bbc_data.getWordVectors(aIndex=aIndexes,topK=k)
            word_vectors = [ temp for temp in word_vectors if len(temp)>=k]
            word_vectors = np.concatenate(word_vectors,axis=1).tolist()

            initial_data = [['TRUE', m, n, k, timetaken]]

            with open(output_file,'w') as fp:
                writer = csv.writer(fp)
                writer.writerows(initial_data)
                writer.writerows(word_vectors)
