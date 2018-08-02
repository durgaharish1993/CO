from cvxpy import *
import numpy as np
from AlgoData import Data
from multiprocessing import Pool
from itertools import product





class Xray(Data):
    def __init__(self, all_data):

        x_given = all_data['x_given']
        w_given = all_data['w_given']
        h_given = all_data['h_given']
        n_given = all_data['n_given']
        r_given = all_data['r_given']
        m_given = all_data['m_given']
        r_expected = all_data['r_expected']
        EXT_POINT_SELECTION = all_data['ext_selection']
        RANDOM_SEED   = all_data['random_seed']
        PROCESSES     = all_data['processes']
        PARALLEL_MODE  = all_data['parallel_mode']


        # This is for calling the super class
        Data.__init__(self, x_given, w_given, h_given, n_given, m_given, r_given )
        self.X = x_given
        self.X_shape = self.X.shape
        self.A = []

        '''
        These are constants for the algorithm

        R:                   This is for choosing the number of anchor variables.
        EXT_POINT_SELECTION: This is for exterior point selection options are "rand", "max", "dist"

        '''
        self.EXT_POINT_SELECTION = EXT_POINT_SELECTION
        self.RANDOM_SEED  = RANDOM_SEED
        self.r = r_expected
        self.FINAL_W_SHAPE = (self.X_shape[0], self.r)
        self.FINAL_H_SHAPE = (self.r, self.X_shape[1])
        self.m = self.X_shape[0]
        self.n = self.X_shape[1]
        self.processes = PROCESSES
        self.parllel_mode = PARALLEL_MODE
        self.exFunSelection = {
            'rand': self._rand_exterior,
            'max': self._max_exterior,
            'dist': self._dist_exterior
        }

        # Asserting : to check everything is good to start the algorithm
        assert self.r < self.X_shape[1], "The Shape of R is not Greater"
        assert isinstance(self.X, np.matrix), "The type is not matrix "
        # assert not np.any(self.X < 0), "X is not a positive matrix."

    def run_algoritm_synthetic(self):

        self.initializeAlgo()
        iteration = 0
        while len(self.A) < self.r and iteration < 100:
            print("##############################################################")
            print('Iteration Number : ', iteration)

            # try:

            self.detection_step()
            self.update_A()
            print(self.A)
            if self.parllel_mode:

                self.projection_step_parallel()
            else:
                h_matrix = self.projection_step()


            #h_matrix = self.projectionStep_Fast()
            self.update_Residual()
            iteration += 1

        if self.r == self.r_given:
            self._strResult_2()

        return sorted(self.A)

    def initializeAlgo(self):
        self.R_matrix = self.X
        self.p = np.matrix(np.ones((self.m, 1)))
        if self.RANDOM_SEED:
            np.random.seed(1)


    def detection_step(self):

        i = self.exFunSelection[self.EXT_POINT_SELECTION]()

        best_val = 0
        list_best_ind = []
        best_ind = None
        for j in range(self.n):
            cur_val = ((self.R_matrix[:, i].T) * (self.X[:, j])) / (self.p.T * self.X[:, j])
            cur_val = cur_val[0,0]
            if cur_val >= best_val:
                if cur_val > best_val:
                    list_best_ind=[j]
                    best_val = cur_val
                    best_ind = j
                else:
                    best_ind = j
                    list_best_ind+=[j]
                    best_val = cur_val

        print('THE BEST INDICES ARE:', list_best_ind)
        self.best_j = best_ind

    def update_A(self):
        if (self.best_j not in self.A) and (self.best_j != None):
            self.A += [self.best_j]

        else:
            print("Nothing Can be added")






    def projection_step_parallel(self):
        self.Xa = self._calculate_XA()
        pool = Pool(processes=self.processes)


        parallel_data=[]
        for i in range(self.n):
            parallel_data += [(self.X[:,i],self.Xa,i)]

        x_result = pool.map(self._Parallel_Regression, parallel_data)
        h_matrix = np.concatenate(x_result,axis=1)
        self.H_matrix = h_matrix





    def getLossValue(self):
        W_obtained = self.X[:,sorted(self.A)]
        if self.W_given.shape == W_obtained.shape:

            self.loss = np.linalg.norm(self.W_given - W_obtained, ord='fro')
        else:
            self.loss = 'None'

        return self.loss







    @staticmethod
    def _Parallel_Regression(data):

        x             = data[0]
        xa            = data[1]
        i             = data[2]
        (m,cur_r)     = xa.shape
        h_i           = Variable(cur_r)
        constraints   = [h_i>=0]
        obj           = Minimize( norm(x-(xa*h_i))  )
        prob          = Problem(obj,constraints)
        prob.solve()
        #print('Objective Value :', i, prob.value, h_i.value)

        h_i_matrix    = np.matrix(h_i.value)
        return h_i_matrix






    def projection_step(self):
        print("This is Projection Step.")
        self.Xa = self._calculate_XA()
        cur_m = self.Xa.shape[1]

        H = Variable(cur_m, self.n)
        constraint = [H >= 0]
        obj = Minimize(norm(self.X - (self.Xa * H), 'fro'))
        prob = Problem(obj, constraint)
        print("Solving the Optimization Problem.")
        prob.solve(solver=SCS)

        if prob.status != OPTIMAL:
            raise Exception("Solver did not converge!")

        print('Objective Value :', prob.value)
        H_matrix = np.matrix(H.value)
        print("Updating H Matrix")
        self.H_matrix = H_matrix
        return H_matrix


    #Cyclic Coordinate Descent Algorithm
    def projectionStep_Fast(self):
        cur_r = len(self.A)
        self.Xa = self._calculate_XA()
        norm_X  = (np.linalg.norm(self.X, 'fro') ** 2)
        tol = 0.01
        maxcycles = 100000


        if len(self.A) ==1:
            B        = np.matrix(np.random.uniform(low=0, high=1, size=(cur_r,self.n)))   # (r x n)
        else:
            B_Prime  = self.H_matrix
            if self.H_matrix.shape[0]< cur_r:

                B        = np.matrix(np.random.uniform(low=0, high=1, size=(1,self.n)))
                B        = np.concatenate([B_Prime,B],axis=0)
            else:
                B        = self.H_matrix


        C        = self.X.T * self.X                                                  # (n x n)
        assert  C.shape == (self.n,self.n)
        S        = self.Xa.T * self.Xa                                                # (r x r)
        small_S  = S.diagonal()                                                       # (1 x r)
        U        = B.T * S                                                            # (n x r)
        pre_objective = float('-inf')
        iterations = 0
        while True:
            iterations +=1
            for i in range(cur_r):
                b         = B.T[:,i]                                                           # (n x 1)
                delta     = B.T[:,i]                                                           # (n x 1)
                u         = U[:,i] - (small_S[0,i] * b )                                         # (n x 1)
                b         = (self._postive_b_matrix(C[:,i], u)) / small_S[0,i]                   # (n x 1)
                delta     = b - delta                                                          # (n x 1)
                U         = U + (delta * (S[:,i]).T )                                            # (n x r)
                B.T[:,i]  = b

            iter_obje = norm_X
            for i in range(cur_r):
                iter_obje += (((U[:,i] + C[:,i]).T) * ((B.T)[:,i])) [0,0]

            cal_fun = lambda x,y : y-x if x-y < 0 else x-y

            objective_delta = cal_fun(pre_objective,iter_obje)
            pre_objective = iter_obje
            if objective_delta < tol  or iterations > maxcycles:
                print("OBJECTIVE DELTA:",objective_delta)
                break

        self.H_matrix = B

        return B


    def update_Residual(self):
        self.R_matrix = self.X - (self.Xa * self.H_matrix)



    def _postive_b_matrix(self,a,b):

        c = (a-b)
        c[c<0] = 0

        return c

    def _calculate_XA(self):
        return self.X[:, self.A]

    def _rand_exterior(self):
        '''
        This is function selecting random exterior point 
        :return:  Index i
        '''
        checking_postive = lambda x: True if (not np.any(x < 0)) else False
        check_greater = np.apply_along_axis(checking_postive, 0, self.R_matrix)
        get_indices = np.where(check_greater == True)[0]
        i = np.random.choice(get_indices)

        return i



    def _max_exterior(self):
        '''
        This is function selecting max exterior point 
        :return:  Index i
        '''
        i = np.argmax(np.linalg.norm(self.R_matrix, axis=0))
        print('The Maximum Index is:', i)
        return i

    def _dist_exterior(self):
        '''
        This is function selecting dist exterior point 
        :return:  Index i
        '''
        m,n =  self.R_matrix.shape
        best_nor = float('-inf')
        for i in range(n):
            temp_mat = (self.R_matrix[:,i].T * self.X)
            temp_mat[temp_mat<0] = 0
            cur_nor = np.linalg.norm(temp_mat)
            if cur_nor> best_nor:
                best_i   = i
                best_nor = cur_nor

        return best_i




    def _strResult_1(self):
        print('----------------------The Algorithm is Done ')
        print('The Matrix W Obtained:\n', self.Xa)
        print('The Matrix W Given:\n', self.W_given)

        print('The Matrix H Obtained:\n', self.H_matrix)
        print('The Matrix H Given:\n', self.H_given)

    def _strResult_2(self):
        print("--------------------------------------------------------")
        print("Final Anchors obtained are:", sorted(self.A))
        # print("The Difference in Norm of W :", round(np.linalg.norm(np.linalg.norm(self.Xa - self.W_given,axis=0)),3))
        # print("The Difference in Norm of H:",  round(np.linalg.norm(np.linalg.norm(self.H_matrix - self.H_given,axis=0)),3))
        # print("The Identity Matrix Obtained:\n", np.multiply(self.H_matrix[:,:self.r],np.eye(self.r)))


