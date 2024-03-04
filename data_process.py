from re import S
from torch._C import dtype
from nltk.tokenize import word_tokenize
import numpy as np
import random
import torch
import json

#import nltk
#nltk.download('punkt')

class DataProcess():
    def __init__(self, file1, file2, file3, file4, file5, file6):
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.file4 = file4
        self.file5 = file5
        self.file6 = file6

        self.news_id = {'NULL': 0}  # {'NULL': 0, 'N46466': 1}
        self.title_content = {}  # {'N46466': ['the', 'brands', 'queen', 'elizabeth', ',', 'prince', 'charles', ',', 'and', 'prince', 'philip', 'swear', 'by']}
        self.abstract_content = {}
        self.word_dict = {'PADDING': 0}
        self.entity_dict = {'PADDING': 0}
        self.news_title_dict = {'0': [0] * 20}  # {'1': [1, 2, ……, 30]}
        self.news_abstract_dict = {'0': [0] * 40}
        self.news_entity_dict = {'0': [0] * 5}
        self.newsid_topic = {0: 'NULL'}
        self.entity_news = {}
        self.entity_matrix_dict = {0: np.zeros(100, dtype='float32')}
        self.embedding_dict = {}

        self.userid_dict = {'NULL': 0}
        self.npratio1 = 4
        self.npratio2 = 50
        self.train_candidate = []
        self.train_label = []
        self.train_user_his = []
        self.user_his_pad = {0: [0] * 50, }
        self.user_his_complete = {0: [], }
        
        self.val_index = []
        self.val_candidate = []
        self.val_label = []
        self.val_user_his = []
        self.val_user = []

        self.test_index = []
        self.test_candidate = []
        self.test_label = []
        self.test_user_his = []
        self.test_user = []
    
    def newsample(self, array, npratio):
        if npratio > len(array):
            return random.sample(array*(npratio // len(array) + 1), npratio)
        else:
            return random.sample(array, npratio)


    # 处理新闻数据

    def process_news(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            self.title_content[line[0]] = word_tokenize((line[3]).lower())
            self.abstract_content[line[0]] = word_tokenize((line[4]).lower())
            if line[0] not in self.news_id:
                self.news_id[line[0]] = len(self.news_id)
            
            title = []
            for word in self.title_content[line[0]]:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                title.append(self.word_dict[word])
            title = title[:20]
            if self.news_id[line[0]] not in self.news_title_dict:
                self.news_title_dict[self.news_id[line[0]]] = title + [0] * (20 - len(title))

            abstract = []
            for word in self.abstract_content[line[0]]:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                abstract.append(self.word_dict[word])
            abstract = abstract[:40]
            if self.news_id[line[0]] not in self.news_abstract_dict:
                self.news_abstract_dict[self.news_id[line[0]]] = abstract + [0] * (40 - len(abstract))
            
            entity = []
            for d in json.loads(line[6]):
                if d['WikidataId'] not in self.entity_dict:
                    self.entity_dict[d['WikidataId']] = len(self.entity_dict)
                entity.append(self.entity_dict[d['WikidataId']])
            for d in json.loads(line[7]):
                if d['WikidataId'] not in self.entity_dict:
                    self.entity_dict[d['WikidataId']] = len(self.entity_dict)
                entity.append(self.entity_dict[d['WikidataId']])
            entity = entity[:5]
            if self.news_id[line[0]] not in self.news_entity_dict:
                self.news_entity_dict[self.news_id[line[0]]] = entity + [0] * (5 - len(entity))

    def generate_entity_matrix(self):
        print ('generate entity matrix start')
        entity_embed = {}
        f1 = open('/home/wangshicheng/news_recommendation/MINDlarge_train/entity_embedding.vec', 'r')
        lines1 = f1.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if line[0] not in self.entity_dict:
                self.entity_dict[line[0]] = len(self.entity_dict)
            if self.entity_dict[line[0]] not in entity_embed:
                entity_embed[self.entity_dict[line[0]]] = np.array([float(i) for i in line[1:]])
        f2 = open('/home/wangshicheng/news_recommendation/MINDlarge_dev/entity_embedding.vec', 'r')
        lines2 = f2.readlines()
        for line in lines2:
            line = line.strip().split('\t')
            if line[0] not in self.entity_dict:
                self.entity_dict[line[0]] = len(self.entity_dict)
            if self.entity_dict[line[0]] not in entity_embed:
                entity_embed[self.entity_dict[line[0]]] = np.array([float(i) for i in line[1:]])
        
        self.entity_matrix = [0] * len(self.entity_dict)
        for k,v in self.entity_dict.items():
            if k in entity_embed:
                self.entity_matrix_dict[k] = entity_embed[k]
            else:
                self.entity_matrix_dict[k] = np.zeros(100, dtype='float32')
        
        self.entity_matrix = torch.FloatTensor(np.array(list(self.entity_matrix_dict.values()), dtype = 'float32'))
        print ('generate entity matrix finished')
        return self.entity_matrix


    def process_train_val_news(self):
        print ('process news start')
        self.process_news(self.file1)
        self.process_news(self.file2)
        self.news_title = np.array(list(self.news_title_dict.values()), dtype = 'int32')
        self.news_abstract = np.array(list(self.news_abstract_dict.values()), dtype = 'int32')
        self.news_entity = np.array(list(self.news_entity_dict.values()), dtype = 'int32')
        print ('news_title.shape: ', self.news_title.shape, 'news_abstract.shape: ', self.news_abstract.shape)
        print ('process news finished')
        #return self.news_title, self.news_abstract, self.news_entity
        #return self.news_title, self.news_entity
        return self.news_title, self.news_abstract

    def generate_user_his(self):
        f3 = open(self.file3)
        lines = f3.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            click_his_complete = [self.news_id[index] for index in line[3].split()]
            click_his_pad = [self.news_id[index] for index in line[3].split()][:50]
            click_his_pad = click_his_pad + [0] * (50 - len(click_his_pad))

            if line[1] not in self.userid_dict:
                self.userid_dict[line[1]] = len(self.userid_dict)

            if self.userid_dict[line[1]] not in self.user_his_pad:
                self.user_his_pad[self.userid_dict[line[1]]] = click_his_pad
                self.user_his_complete[self.userid_dict[line[1]]] = click_his_complete
        f3.close()

        f4 = open(self.file4)
        lines = f4.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            click_his_complete = [self.news_id[index] for index in line[3].split()]
            click_his_pad = [self.news_id[index] for index in line[3].split()][:50]
            click_his_pad = click_his_pad + [0] * (50 - len(click_his_pad))

            if line[1] not in self.userid_dict:
                self.userid_dict[line[1]] = len(self.userid_dict)

            if self.userid_dict[line[1]] not in self.user_his_pad:
                self.user_his_pad[self.userid_dict[line[1]]] = click_his_pad
                self.user_his_complete[self.userid_dict[line[1]]] = click_his_complete
        f4.close()
        return self.user_his_pad

    # 处理训练集数据
    def pre_train_behaviors(self):
        print ('reset train variables')
        self.train_candidate = []
        self.train_label = []
        self.train_user_his = []
        self.train_user = []
        
        print ('process train behaviors start')

        f3 = open(self.file3)
        lines = f3.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            p_doc, n_doc = [], []
            for i in line[4].split():
                if int(i.split('-')[1]) == 1:
                    p_doc.append(self.news_id[i.split('-')[0]])
                elif int(i.split('-')[1]) == 0:
                    n_doc.append(self.news_id[i.split('-')[0]])

            for doc in p_doc:
                neg_doc = self.newsample(n_doc, self.npratio1)
                neg_doc.append(doc)
                candidate_label = [0] * self.npratio1 + [1]
                candidate_order = list(range(self.npratio1 + 1))
                random.shuffle(candidate_order)
                candidate_shuffle = []
                candidate_label_shuffle = []
                for i in candidate_order:
                    candidate_shuffle.append(neg_doc[i])
                    candidate_label_shuffle.append(candidate_label[i])
                self.train_candidate.append(candidate_shuffle)
                self.train_label.append(candidate_label_shuffle)
                self.train_user.append(self.userid_dict[line[1]])
        self.train_candidate = torch.LongTensor(np.array(self.train_candidate, dtype='int32'))
        self.train_label = torch.FloatTensor(np.array(self.train_label, dtype='int32'))
        self.train_user = torch.LongTensor(np.array(self.train_user, dtype = 'int32'))

        print ('train_candidate.size: ', self.train_candidate.size())
        print ('train_label.size:', self.train_label.size())
        print ('train_user.size:', self.train_user.size())
        print ('process train behaviors finished')
        return [self.train_candidate, self.train_user, self.train_label]


    # 处理验证集数据
    def pre_val_behaviors(self, file):
        print('process val behaviors start')

        f4 = open(file)
        lines = f4.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            p_doc, n_doc = [], []
            for i in line[4].split():
                if int(i.split('-')[1]) == 1:
                    p_doc.append(self.news_id[i.split('-')[0]])
                elif int(i.split('-')[1]) == 0:
                    n_doc.append(self.news_id[i.split('-')[0]])

            sess_index = []
            sess_index.append(len(self.val_candidate))
            for i in p_doc:
                self.val_candidate.append(i)
                self.val_label.append(1)
                self.val_user.append(self.userid_dict[line[1]])
            
            for i in n_doc:
                self.val_candidate.append(i)
                self.val_label.append(0)
                self.val_user.append(self.userid_dict[line[1]])

            sess_index.append(len(self.val_candidate))
            self.val_index.append(sess_index)

        self.val_candidate = np.array(self.val_candidate, dtype='int32')
        self.val_label = torch.FloatTensor(np.array(self.val_label, dtype='int32'))
        self.val_user = torch.LongTensor(np.array(self.val_user, dtype = 'int32'))

        print('val_candidate.shape: ', self.val_candidate.shape)
        print('val_label.size: ', self.val_label.size())
        print ('val_user.size:', self.val_user.size())
        print('len(val_index): ', len(self.val_index))

        print('process val behaviors finished')
        return [self.val_candidate, self.val_user, self.val_label, self.val_index]

    def pre_test_behaviors(self, file):
        print('process test behaviors start')

        f4 = open(file)
        lines = f4.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            p_doc, n_doc = [], []
            for i in line[4].split():
                p_doc.append(self.news_id[i.split('-')[0]])

            sess_index = []
            sess_index.append(len(self.test_candidate))
            for i in p_doc:
                self.test_candidate.append(i)
                self.test_label.append(1)
                self.test_user.append(self.userid_dict[line[1]])

            sess_index.append(len(self.test_candidate))
            self.test_index.append(sess_index)

        self.test_candidate = np.array(self.test_candidate, dtype='int32')
        self.test_label = torch.FloatTensor(np.array(self.test_label, dtype='int32'))
        self.test_user = torch.LongTensor(np.array(self.test_user, dtype = 'int32'))

        print('test_candidate.shape: ', self.test_candidate.shape)
        print('test_label.size: ', self.test_label.size())
        print ('test_user.size:', self.test_user.size())
        print('len(test_index): ', len(self.test_index))

        print('process test behaviors finished')
        return [self.test_candidate, self.test_user, self.test_label, self.test_index]
            

    # 加载glove预训练模型
    def load_glove(self):
        print ('load glove start')

        f = open(self.file5)
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                break
            line = line.strip().split()
            if len(line) != 301:
                continue
            word = line[0].encode('utf-8').decode()
            if word not in self.word_dict:
                continue
            if len(word) != 0:
                vec = [float(x) for x in line[1:]]
                self.embedding_dict[word] = vec

        self.embedding_matrix = [0] * len(self.word_dict)
        cand = []
        for k, v in self.embedding_dict.items():
            self.embedding_matrix[self.word_dict[k]] = np.array(v, dtype='float32')
            cand.append(self.embedding_matrix[self.word_dict[k]])

        cand = np.array(cand, dtype='float32')
        mu = np.mean(cand, axis=0)
        Sigma = np.cov(cand.T)
        norm = np.random.multivariate_normal(mu, Sigma, 1)
        for i in range(len(self.embedding_matrix)):
            if type(self.embedding_matrix[i]) == int:
                self.embedding_matrix[i] = np.reshape(norm, 300)

        self.embedding_matrix[0] = np.zeros(300, dtype='float32')
        self.embedding_matrix = torch.FloatTensor(np.array(self.embedding_matrix, dtype='float32'))

        print('embedding_matrix.size: ', self.embedding_matrix.size())
        print('load glove process finished')

        return self.embedding_matrix

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np

if __name__ == '__main__':
    
    file1 = '/home/wangshicheng/news_recommendation/MINDlarge_train/news.tsv'
    file2 = '/home/wangshicheng/news_recommendation/MINDlarge_dev/news.tsv'
    file3 = '/home/wangshicheng/news_recommendation/MINDlarge_train/behaviors.tsv'
    file4 = '/home/wangshicheng/news_recommendation/MINDlarge_dev/behaviors.tsv'
    file5 = '/home/wangshicheng/news_recommendation/glove.840B.300d.txt'

    data_module = DataProcess(file1, file2, file3, file4, file5)
    news_title, news_abstract = data_module.process_train_val_news()
    news_title, news_abstract = torch.LongTensor(news_title), torch.LongTensor(news_abstract)

    [train_candidate, train_label, train_user] = data_module.pre_train_behaviors()
    train_dataset = Data.TensorDataset(train_candidate, train_label, train_user)
    train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 900, shuffle = True, num_workers = 2)

    [val_candidate,val_user, val_label, val_index] = data_module.pre_val_behaviors()
    
    user_his_complete = data_module.user_his_complete
    user_his_pad = torch.LongTensor(np.array(list(data_module.user_his_pad.values()), dtype = 'int32'))
    print (user_his_pad.size(), len(user_his_complete))

    data, user, news = [], [], []
    for u, h in user_his_complete.items():
        for n in h:
            data.append(1)
            user.append(u)
            news.append(n)
    print (len(data), len(data_module.userid_dict), len(data_module.news_id))

    import scipy.sparse as sp
    u_n = sp.csr_matrix((data, (user, news)), shape = (len(data_module.userid_dict), len(data_module.news_id)))
    #print (spm)

    u_u = sp.coo_matrix(u_n.dot(u_n.transpose()))
    #sp.save_npz('./large_user_adj.npz', u_u)
    
    import pickle
    #user_adj = sp.load_npz('./large_user_adj.npz')
    data, row, col = u_u.data, u_u.row, u_u.col
    print (user_adj.shape, type(data), type(row), len(row))
    user_adj_dic = {}
    user_adj = {0: [0] * 3}

    for i in range(len(row)):
        if row[i] not in user_adj_dic:
            user_adj_dic[row[i]] = []
        user_adj_dic[row[i]].append(col[i])
    for k, v in user_adj_dic.items():
        if len(v) > 3:
            user_adj[k] = random.sample(v, 3)
        else:
            user_adj[k] = v + [0] * (3 - len(v))
    user_adj = np.array(list(user_adj.values()), dtype = 'int32')
    print (len(user_adj_dic), user_adj.shape)

    f1 = open('large_user_adj.pkl', 'wb')
    pickle.dump(user_adj, f1)
    f1.close()
    
    f1 = open('large_user_adj.pkl',  'rb')
    user_adj_load = pickle.load(f1)
    user_adj_load = torch.LongTensor(user_adj_load)
    print (user_adj_load.size())
'''