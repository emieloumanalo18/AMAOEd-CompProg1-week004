import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt

settings= {
    'n_dimesnion': 100,
    'epochs' : 5000,
    'learning_rate' : 0.01
    }



class NueralNetwork():
    def __init__ (self):
        self.n = settings['n_dimesnion']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        pass
    
    
    
    def generate_training_data(self, data_pairs):
        pair_counts = defaultdict(int)
        x_ = []
        y_ = []

        for i in range(len(data_pairs)):

            x_.append(data_pairs[i][0])

            for product in x_:
                pair_counts[product] +=1

            y_.append(data_pairs[i][1])

            y = self.tokenizer(y_)

            for contents in y:
                for content in contents:
                    pair_counts[content] +=1

        self.pairs_len = len(pair_counts.keys())

        self.pairs_list = sorted(list(pair_counts.keys()), reverse=False)
        self.pairs_index = {x:i for (i, x) in enumerate(self.pairs_list)}
        self.index_pairs = {i:x for (i, x) in enumerate(self.pairs_list)}

        
        train_data = self.batch(data_pairs)
        training_data = []
        
        for i in range(len(train_data)):
            for i, (X, Y) in enumerate(train_data):
                target = self.pairs_index[X] 
                w_target = [0 for i in range(0, self.pairs_len)]
                w_target[target] = 1

                w_context = []
                for contexts in y[i]:
                    content = self.pairs_index[contexts]
                    w_content = [0 for i in range(0, self.pairs_len)]
                    w_content[content] = 1

                    w_context.append(w_content)

                training_data.append([w_target, w_context])

        return np.array(training_data, dtype=object)
    
    
    
    
    def batch(self, data_pairs):
        bt = np.random.choice(np.shape(data_pairs)[0], size=50)

        train_data =[]
        for i in bt:
            train_data.append(data_pairs[i])

        return train_data  



    def tokenizer(self, words):
        token = []
        for word in words:
            token.append(word.split())

        return token 



    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    

        

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
                

        
        
    def backprop(self, e, h, x): 
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        
    
        

    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (self.pairs_len, self.n))    
        self.w2 = np.random.uniform(-1, 1, (self.n, self.pairs_len))   
             
                  
        for i in range(self.epochs):
            self.loss = 0
            for c, (w_t, w_c) in enumerate(training_data):
                y_pred, h, u = self.forward_pass(w_t)
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(EI, h, w_t)

                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
            if i % 100 == 0:  
                print('Iteration: ',i, 'Loss: ', self.loss)
        
    
    
    

    def word_vec(self, word):
        w_index = self.pairs_index[word]
        v_w = self.w1[w_index]
        return v_w
    
    
    
    
    def vec_sim(self, word, top_n=10):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.pairs_len):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
    
            word = self.pairs_index[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
        
        get_word = []
        get_sim = []
        for word, sim in words_sorted[:top_n]:
            get_word.append(word)
#             get_sim.append(sim)
            
        return get_word
 
     
      
#     def recommendation(dataset, product_name): 
#         sim_products = vec_sim(product_name)
           
#         get data in row with index of sim_products using for loop
           

nn = NueralNetwork()



