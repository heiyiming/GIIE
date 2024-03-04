from re import S
import re
import torch
from torch.functional import tensordot
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ScaledAttention(nn.Module):
	def __init__(self, temperature, dropout_rate):
		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, q, k, v):
		score = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
		score = F.softmax(score, dim=-1)
		score = self.dropout(score)
		output = torch.matmul(score, v)
		return score, output


class MultiHeadAttention(nn.Module):
	def __init__(self, num_head, embedding_dim, hid_dim, dropout_rate):
		super().__init__()
		self.num_head = num_head
		self.size_per_head = hid_dim // num_head
		self.hid_dim = hid_dim

		self.q_linear = nn.Linear(embedding_dim, hid_dim)
		self.k_linear = nn.Linear(embedding_dim, hid_dim)
		self.v_linear = nn.Linear(embedding_dim, hid_dim)
		self.fc = nn.Linear(hid_dim, hid_dim)
		self.dropout = nn.Dropout(dropout_rate)

		self.attention = ScaledAttention(temperature = self.size_per_head ** 0.5, dropout_rate = dropout_rate)
	
	def forward(self, q, k, v):  # [30, 10, 50, 400]
		sample_size = q.size()[0]
		batch_size = q.size()[1]
		q_len, k_len, v_len = q.size()[2], k.size()[2], v.size()[2]

		q = self.q_linear(q).view(sample_size, batch_size, q_len, self.num_head, self.size_per_head)
		k = self.k_linear(k).view(sample_size, batch_size, k_len, self.num_head, self.size_per_head)
		v = self.v_linear(v).view(sample_size, batch_size, v_len, self.num_head, self.size_per_head)

		q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
		score, output = self.attention(q, k, v)

		output = output.transpose(2, 3).contiguous().view(sample_size, batch_size, v_len, self.hid_dim)
		output = self.fc(output)
		output = self.dropout(output)
		return output

'''
class News_Encoder(nn.Module):
	def __init__(self, num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, dropout_rate):
		super().__init__()
		self.word_embedding = nn.Embedding.from_pretrained(word_matrix, freeze = False)
		self.entity_embedding = nn.Embedding.from_pretrained(entity_matrix, freeze = False)
		self.word_attention = MultiHeadAttention(num_head, word_dim, hid_dim, dropout_rate)
		self.entity_attention = MultiHeadAttention(num_head, entity_dim, hid_dim, dropout_rate)
		self.W = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(200, 1))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj.data, gain=1.414)
		self.dropout = nn.Dropout(dropout_rate)

		self.w1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w3 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.w1, gain = 1.414)
		nn.init.xavier_uniform_(self.w2, gain = 1.414)
		nn.init.xavier_uniform_(self.w3, gain = 1.414)

	def forward(self, title, abstract):
		title_size = title.size()
		title = self.word_embedding(title)
		title = self.word_attention(title, title, title)
		title_att = torch.tanh(torch.matmul(title, self.W))
		title_att = torch.matmul(title_att, self.proj)
		title_att = F.softmax(title_att, dim = 2)
		title = torch.matmul(title_att.transpose(-2, -1), title).squeeze(dim = 2)

		abstract = self.word_embedding(abstract)
		abstract = self.word_attention(abstract, abstract, abstract)
		abstract_att = torch.tanh(torch.matmul(abstract, self.W))
		abstract_att = torch.matmul(abstract_att, self.proj)
		abstract_att = F.softmax(abstract_att, dim = 2)
		abstract = torch.matmul(abstract_att.transpose(-2, -1), abstract).squeeze(dim = 2)
	   
		news_rep = torch.matmul(title, self.w1) + torch.matmul(abstract, self.w2)
		return news_rep.reshape(title_size[0], title_size[1], -1)
'''

class News_Encoder(nn.Module):
	def __init__(self, num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, dropout_rate):
		super().__init__()
		self.word_embedding = nn.Embedding.from_pretrained(word_matrix, freeze = True)
		self.entity_embedding = nn.Embedding.from_pretrained(entity_matrix, freeze = False)
		self.word_attention = MultiHeadAttention(num_head, word_dim, hid_dim, dropout_rate)
		self.entity_attention = MultiHeadAttention(num_head, entity_dim, hid_dim, dropout_rate)
		self.hid_dim = hid_dim
		
		self.W1 = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj1 = nn.Parameter(torch.Tensor(200, 1))
		self.W2 = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj2 = nn.Parameter(torch.Tensor(200, 1))
		self.W_agg = nn.Parameter(torch.Tensor(hid_dim, 1))
		nn.init.xavier_uniform_(self.W1.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj1.data, gain=1.414)
		nn.init.xavier_uniform_(self.W2.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj2.data, gain=1.414)
		nn.init.xavier_uniform_(self.W_agg.data, gain=1.414)
		
		self.dropout = nn.Dropout(dropout_rate)

		self.w1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w3 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.w1.data, gain = 1.414)
		nn.init.xavier_uniform_(self.w2.data, gain = 1.414)
		nn.init.xavier_uniform_(self.w3.data, gain = 1.414)

	def forward(self, title, abstract):
		title_size = title.size() # [30, 5, 30]
		title = self.word_embedding(title)
		title = self.word_attention(title, title, title)
		title_att = torch.tanh(torch.matmul(title, self.W1))   # [30, 5, 30, 400]
		title_att = torch.matmul(title_att, self.proj1)
		title_att = F.softmax(title_att, dim = 2)
		title = torch.matmul(title_att.transpose(-2, -1), title).squeeze(dim = 2)	 # [30, 5, 400]

		abstract = self.word_embedding(abstract)
		abstract = self.word_attention(abstract, abstract, abstract)
		abstract_att = torch.tanh(torch.matmul(abstract, self.W2))
		abstract_att = torch.matmul(abstract_att, self.proj2)
		abstract_att = F.softmax(abstract_att, dim = 2)
		abstract = torch.matmul(abstract_att.transpose(-2, -1), abstract).squeeze(dim = 2)
	   
		#news_rep = torch.matmul(title, self.w1) + torch.matmul(abstract, self.w2)
		
		news_rep = torch.cat((title.reshape(-1, self.hid_dim).unsqueeze(dim = 1), abstract.reshape(-1, self.hid_dim).unsqueeze(dim = 1)), dim = 1)	  # [150, 2, 400]
		att = torch.tanh(torch.matmul(news_rep, self.W_agg))
		att = F.softmax(att, dim = 1)
		news_rep = torch.matmul(att.transpose(-1, -2), news_rep).squeeze(dim = 1).reshape(title_size[0], title_size[1], self.hid_dim)

			
		return news_rep.reshape(title_size[0], title_size[1], -1)

class Multi_Rep_Encoder(nn.Module):
	def __init__(self, hid_dim, num_prototype, multi_rep_mode, dropout_rate):
		super().__init__()
		self.hid_dim = hid_dim
		self.num_prototype = num_prototype
		self.mode = multi_rep_mode
		self.dropout = nn.Dropout(dropout_rate)

		self.prototype = nn.Parameter(torch.Tensor(num_prototype, hid_dim))
		self.w1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.W = nn.Parameter(torch.Tensor(num_prototype, hid_dim, hid_dim))
		self.w = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(200, 1))

		nn.init.xavier_uniform_(self.prototype, gain=1.414)
		nn.init.xavier_uniform_(self.w1.data, gain=1.414)
		nn.init.xavier_uniform_(self.w2.data, gain=1.414)
		nn.init.xavier_uniform_(self.W.data, gain = 1.414)
		nn.init.xavier_uniform_(self.w.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj.data, gain = 1.414)

	def forward(self, news_rep):
		news_rep_size = news_rep.size()    # [30, 5, 400]
		news_rep = news_rep.reshape(-1, news_rep_size[-1])

		if self.mode == 'concat':
			news_rep = news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1)
			news_rep = torch.matmul(news_rep, self.w1) + torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep_size[0] * news_rep_size[1], 1, 1), self.w2)
			#news_rep = news_rep + torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep_size[0] * news_rep_size[1], 1, 1), self.w2)
			news_rep = self.dropout(news_rep)
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'att':
			news_rep = news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1)   # [150, 10, 400]
			prototype = self.prototype.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1)
			news_rep = torch.cat((news_rep.unsqueeze(dim = 2), prototype.unsqueeze(dim = 2)), dim = 2)
			att = torch.tanh(torch.matmul(news_rep, self.w))
			att = torch.matmul(att, self.proj)
			att = F.softmax(att, dim = 2)	  # 
			news_rep = torch.matmul(att.transpose(-1, -2), news_rep).squeeze(dim = 2)
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'cat':
			news_rep = news_rep.unsqueeze(dim = 0).repeat(self.num_prototype, 1, 1)
			prototype = self.prototype.unsqueeze(dim = 1).repeat(1, news_rep.size(1), 1)
			news_rep = torch.matmul(news_rep, self.w1) + torch.matmul(prototype, self.w2)
			news_rep = news_rep.reshape(self.num_prototype, news_rep_size[0], news_rep_size[1], -1)
			return news_rep
		
		elif self.mode == 'trans':
			news_rep = news_rep.unsqueeze(dim = 0).repeat(self.num_prototype, 1, 1) # [10, 150, 400], [10, 400, 400]
			news_rep = torch.matmul(news_rep, self.W) # [10, 150, 400]
			news_rep = news_rep.transpose(0, 1).reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, self.hid_dim) #[30, 5, 10, 400]
		
		elif self.mode == 'single':
			news_rep = self.dropout(torch.tanh(news_rep))

		elif self.mode == 'cln_cat':
			multi_news_rep = news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1)
			multi_news_rep = torch.matmul(multi_news_rep, self.w1) + torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep_size[0] * news_rep_size[1], 1, 1), self.w2)
			weight = F.softmax(torch.matmul(news_rep, self.prototype.transpose(-1, -2)), dim = -1)	  # [150, 10]
			news_rep = torch.mul(multi_news_rep, weight.unsqueeze(dim = 2).repeat(1, 1, news_rep.size(-1)))
			news_rep = self.dropout(torch.tanh(news_rep))
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'cln_1':
			weight = torch.tanh(torch.matmul(self.prototype, self.w1))	  # [5, 400]
			news_rep = torch.mul(weight.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1), news_rep.unsqueeze(dim = 1))
			#news_rep = news_rep + torch.matmul(self.prototype, self.w2).unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1)
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'cln':
			weight = torch.matmul(self.prototype, self.w1)	  # [5, 400]
			news_rep = torch.mul(weight.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1), news_rep.unsqueeze(dim = 1))
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep
		
		else:
			weight = torch.sigmoid(torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1), self.w1) + torch.matmul(news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1), self.w2))
			news_rep = torch.mul(weight, news_rep.unsqueeze(dim = 1))
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep


class Multi_Rep_User_Encoder(nn.Module):
	def __init__(self, num_head, hid_dim, dropout_rate, num_prototype):
		super().__init__()
		self.attention = MultiHeadAttention(num_head, hid_dim, hid_dim, dropout_rate)
		#self.attention = ScaledAttention(temperature = 1.0)
		self.W = nn.Parameter(torch.Tensor(num_prototype, hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(num_prototype, 200, 1))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj.data, gain=1.414)
		self.hid_dim = hid_dim
		
		self.dropout = nn.Dropout(dropout_rate)
		

	def forward(self, history_rep):    # [30, 50, 10, 400], [10, 30, 50, 400]
		history_rep = history_rep.transpose(1, 2)	 # [30, 10, 50, 400]
		#history_rep = self.attention(history_rep, history_rep, history_rep)
		#history_rep_size = history_rep.size()
		#history_rep = history_rep.reshape(history_rep_size[0], -1, self.hid_dim)

		att = torch.tanh(torch.matmul(history_rep, self.W))
		att = torch.matmul(att, self.proj)
		#att = att.reshape(history_rep_size[0], history_rep_size[1], history_rep_size[2], -1)
		att = F.softmax(att, dim = 2)
		
		#multi_user_rep = self.dropout(torch.matmul(att.transpose(-2, -1), history_rep.reshape(history_rep_size[0], history_rep_size[1], history_rep_size[2], -1)).squeeze(dim = 2))
		#return multi_user_rep.transpose(0, 1)
		multi_user_rep = torch.matmul(att.transpose(-2, -1), history_rep).squeeze(dim = 2)
		return multi_user_rep

class Multi_Rep_Predictor(nn.Module):
	def __init__(self, num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode, gnn_mode, agg_mode):
		super().__init__()
		self.news_encoder = News_Encoder(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, dropout_rate)
		self.attention = MultiHeadAttention(num_head, hid_dim, hid_dim, dropout_rate)
		self.multi_rep_encoder = Multi_Rep_Encoder(hid_dim, num_prototype, multi_rep_mode, dropout_rate)
		self.user_encoder = Multi_Rep_User_Encoder(num_head, hid_dim, dropout_rate, num_prototype)
		self.prototype = self.multi_rep_encoder.prototype
		self.contrastive_mode = contrastive_mode
		self.gnn_mode = gnn_mode
		self.agg_mode = agg_mode
		self.infoNCE = InfoNCE(hid_dim, infonce_mode, self.prototype)
		self.num_prototype = num_prototype
		self.hid_dim = hid_dim
				
		self.W = nn.Parameter(torch.Tensor(2 * hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.W.data, gain = 1.414)

		self.w1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.w1.data, gain = 1.414)
		self.w2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.w2.data, gain = 1.414)
		self.w3 = nn.Parameter(torch.Tensor(num_prototype, hid_dim))
		nn.init.xavier_uniform_(self.w3.data, gain = 1.414)
		self.w4 = nn.Parameter(torch.Tensor(num_prototype, hid_dim, 1))
		nn.init.xavier_uniform_(self.w4.data, gain = 1.414)
		self.w5 = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(200, 1))
		nn.init.xavier_uniform_(self.w5.data, gain = 1.414)
		nn.init.xavier_uniform_(self.proj.data, gain = 1.414)
		#self.wgcn1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		#self.wgcn2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		#self.wgcn3 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		#nn.init.xavier_uniform_(self.wgcn1.data, gain = 1.414)
		#nn.init.xavier_uniform_(self.wgcn2.data, gain = 1.414)
		#nn.init.xavier_uniform_(self.wgcn3.data, gain = 1.414)

	def forward(self, candidate_title, candidate_abstract, his_title, his_abstract, nei1_title, nei1_abstract, nei2_title, nei2_abstract):
		batch_size = candidate_title.size(0)

		candidate_rep = self.news_encoder(candidate_title, candidate_abstract)	  # [30, 5, 400]
		target_his_rep = self.news_encoder(his_title, his_abstract)    # [30, 50, 400]
		target_his_rep = self.attention(target_his_rep.unsqueeze(dim = 0), target_his_rep.unsqueeze(dim = 0), target_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)

		target_his_rep = self.multi_rep_encoder(target_his_rep)    # [30, 50, 10, 400]
		#target_user_rep = torch.mean(target_his_rep, dim = 1)
		target_user_rep = self.user_encoder(target_his_rep)    # [30, 10, 400]
		
		'''		   
		if self.gnn_mode == 'mgat1':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)    # [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			nei2_user_rep = self.user_encoder(nei2_his_rep)

			neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 1), nei1_user_rep.unsqueeze(dim = 1), nei2_user_rep.unsqueeze(dim = 1)), dim = 1)
			#neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 1), nei1_user_rep.unsqueeze(dim = 1)), dim = 1)
			num_neighbor = neighbor_user_rep.size(1)
		
			# [30, 2, 10, 400]
			t = torch.tanh(torch.matmul(target_user_rep.unsqueeze(dim = 1).repeat(1, num_neighbor, 1, 1), self.w1) + torch.matmul(neighbor_user_rep, self.w2))
			t = t.reshape(batch_size * num_neighbor, self.num_prototype, self.hid_dim).unsqueeze(dim = 2)
			att = torch.matmul(t.reshape(-1, t.size(2), t.size(3)), self.w3.unsqueeze(dim = 0).repeat(t.size(0), 1, 1).unsqueeze(dim = 3).reshape(-1, t.size(3), 1)).squeeze(dim = 2)
			att = att.reshape(batch_size, num_neighbor, self.num_prototype).transpose(-1, -2)
		
			att = F.softmax(att, dim = -1)
			att = att.reshape(-1, num_neighbor).unsqueeze(dim = 1)	  # [300, 1, 2]
			neighbor_info = torch.matmul(att, neighbor_user_rep.transpose(1, 2).reshape(-1, num_neighbor, self.hid_dim))
			# [300, 1, 400] 
			neighbor_info = neighbor_info.squeeze(dim = 1).reshape(-1, self.num_prototype, self.hid_dim)	# [30, 10, 400]
			#target_user_rep = target_user_rep + neighbor_info
			target_user_rep = neighbor_info
		

		elif self.gnn_mode == 'mgat2':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			#nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			#nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			#nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)	# [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			#nei2_user_rep = self.user_encoder(nei2_his_rep)
			#target_user_rep, nei1_user_rep, nei2_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1), nei2_user_rep.transpose(0,1)
			target_user_rep, nei1_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1)

			#neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2), nei2_user_rep.unsqueeze(dim = 2)), dim = 2)
			neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2)), dim = 2)
			num_neighbor = neighbor_user_rep.size(2)

			t = torch.matmul(neighbor_user_rep, self.w1) + torch.matmul(target_user_rep.unsqueeze(dim = 2).repeat(1, 1, num_neighbor, 1), self.w2)
			att = torch.tanh(torch.matmul(t, self.w4.unsqueeze(dim = 1).repeat(1, t.size(1), 1, 1)))
			att = F.softmax(att, dim = 2)

			target_user_rep = torch.matmul(att.transpose(-1, -2), neighbor_user_rep).squeeze(dim = 2)
			target_user_rep = target_user_rep.transpose(0, 1)

		elif self.gnn_mode == 'sgat':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)    # [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			nei2_user_rep = self.user_encoder(nei2_his_rep)
			#target_user_rep, nei1_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1)
			target_user_rep, nei1_user_rep, nei2_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1), nei2_user_rep.transpose(0,1)
			
			#neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2)), dim = 2)
			neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2), nei2_user_rep.unsqueeze(dim = 2)), dim = 2)
			num_neighbor = neighbor_user_rep.size(2)

			t = torch.matmul(neighbor_user_rep, self.w1) + torch.matmul(target_user_rep.unsqueeze(dim = 2).repeat(1, 1, num_neighbor, 1), self.w2)
			att = torch.tanh(torch.matmul(t, self.w5))
			att = torch.matmul(att, self.proj)
			att = F.softmax(att, dim = 2)
			target_user_rep = torch.matmul(att.transpose(-1, -2), neighbor_user_rep).squeeze(dim = 2)
			target_user_rep = target_user_rep.transpose(0, 1)


		elif self.gnn_mode == 'sgcn':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			#nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			#nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			#nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)	# [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			#nei2_user_rep = self.user_encoder(nei2_his_rep)
		  
			#target_user_rep, nei1_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1)
			#target_user_rep, nei1_user_rep, nei2_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1), nei2_user_rep.transpose(0,1)
			#target_user_rep = torch.matmul(target_user_rep, self.wgcn1) + torch.matmul(nei1_user_rep, self.wgcn2)
			target_user_rep = torch.matmul(target_user_rep, self.wgcn1) + torch.matmul(nei1_user_rep, self.wgcn2) + torch.matmul(nei2_user_rep, self.wgcn3)
			
		else:
			pass
		'''

		if self.agg_mode == 'mm_hard_max':
			candidate_rep_ = self.multi_rep_encoder(candidate_rep)	  # [30, 5, 10, 400]
			target_user_rep_ = target_user_rep.unsqueeze(dim = 1).repeat(1, candidate_rep.size(1), 1, 1)
			predict_logits = torch.matmul(candidate_rep_.unsqueeze(dim = 3), target_user_rep_.unsqueeze(dim = 4)).squeeze(dim = 3).squeeze(dim = 3)
			predict_logits = torch.max(predict_logits, dim = 2)[0]
		
		elif self.agg_mode == 'mm_hard_mean':
			candidate_rep_ = self.multi_rep_encoder(candidate_rep)	  # [30, 5, 10, 400]
			target_user_rep_ = target_user_rep.unsqueeze(dim = 1).repeat(1, candidate_rep.size(1), 1, 1)
			predict_logits = torch.matmul(candidate_rep_.unsqueeze(dim = 3), target_user_rep_.unsqueeze(dim = 4)).squeeze(dim = 3).squeeze(dim = 3)
			predict_logits = torch.mean(predict_logits, dim = 2)

		elif self.agg_mode == 'repeat_hard':
			candidate_rep_ = candidate_rep.unsqueeze(dim = 2).repeat(1, 1, self.num_prototype, 1)	 # [30, 5, 10, 400]
			target_user_rep_ = target_user_rep.unsqueeze(dim = 1).unsqueeze(dim = 1).repeat(1, candidate_rep.size(1), self.num_prototype, 1)	# [30, 5, 10, 400]]
			predict_logits = torch.matmul(candidate_rep_.unsqueeze(dim = 3), target_user_rep_.unsqueeze(dim = 4)).squeeze(dim = 3).squeeze(dim = 3)
			predict_logits = torch.max(predict_logits, dim = 2)[0]

		elif self.agg_mode == 'sm_hard_max':
			candidate_rep_ = candidate_rep.unsqueeze(dim = 2).repeat(1, 1, self.num_prototype, 1)	 # [30, 5, 10, 400]
			target_user_rep_ = target_user_rep.unsqueeze(dim = 1).repeat(1, candidate_rep.size(1), 1, 1)	# [30, 5, 10, 400]]
			predict_logits = torch.matmul(candidate_rep_.unsqueeze(dim = 3), target_user_rep_.unsqueeze(dim = 4)).squeeze(dim = 3).squeeze(dim = 3)
			predict_logits = torch.max(predict_logits, dim = 2)[0]
		
		elif self.agg_mode == 'sm_hard_mean':
			candidate_rep_ = candidate_rep.unsqueeze(dim = 2).repeat(1, 1, self.num_prototype, 1)	 # [30, 5, 10, 400]
			target_user_rep_ = target_user_rep.unsqueeze(dim = 1).repeat(1, candidate_rep.size(1), 1, 1)	# [30, 5, 10, 400]]
			predict_logits = torch.matmul(candidate_rep_.unsqueeze(dim = 3), target_user_rep_.unsqueeze(dim = 4)).squeeze(dim = 3).squeeze(dim = 3)
			predict_logits = torch.max(predict_logits, dim = 2)

		elif self.agg_mode == 'soft':
			local_att = torch.matmul(candidate_rep, self.prototype.transpose(-2, -1))	 # [30, 5, 10]
			local_att = F.softmax(local_att, dim = 2)
			local_user_rep = torch.matmul(local_att, target_user_rep)	 # [30, 5, 400]

			predict_logits = torch.matmul(candidate_rep.unsqueeze(dim = 2), local_user_rep.unsqueeze(dim = 3))
			predict_logits = predict_logits.reshape(predict_logits.size(0), predict_logits.size(1))    # [30, 5]
		else:
			candidate_rep_ = self.multi_rep_encoder(candidate_rep).reshape(batch_size, candidate_rep.size(1), self.num_prototype * self.hid_dim)	# [30, 5, 4000]
			target_user_rep_ = target_user_rep.reshape(batch_size, self.num_prototype * self.hid_dim).unsqueeze(dim = 2)	# [30, 4000, 1]
			predict_logits = torch.matmul(candidate_rep_, target_user_rep_).squeeze(dim = 2)


		if self.gnn_mode == 'mgat1':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)    # [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			nei2_user_rep = self.user_encoder(nei2_his_rep)

			neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 1), nei1_user_rep.unsqueeze(dim = 1), nei2_user_rep.unsqueeze(dim = 1)), dim = 1)
			#neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 1), nei1_user_rep.unsqueeze(dim = 1)), dim = 1)
			num_neighbor = neighbor_user_rep.size(1)

			# [30, 2, 10, 400]
			t = torch.tanh(torch.matmul(target_user_rep.unsqueeze(dim = 1).repeat(1, num_neighbor, 1, 1), self.w1) + torch.matmul(neighbor_user_rep, self.w2))
			t = t.reshape(batch_size * num_neighbor, self.num_prototype, self.hid_dim).unsqueeze(dim = 2)
			att = torch.matmul(t.reshape(-1, t.size(2), t.size(3)), self.w3.unsqueeze(dim = 0).repeat(t.size(0), 1, 1).unsqueeze(dim = 3).reshape(-1, t.size(3), 1)).squeeze(dim = 2)
			att = att.reshape(batch_size, num_neighbor, self.num_prototype).transpose(-1, -2)

			att = F.softmax(att, dim = -1)
			att = att.reshape(-1, num_neighbor).unsqueeze(dim = 1)	  # [300, 1, 2]
			neighbor_info = torch.matmul(att, neighbor_user_rep.transpose(1, 2).reshape(-1, num_neighbor, self.hid_dim))
			# [300, 1, 400] 
			neighbor_info = neighbor_info.squeeze(dim = 1).reshape(-1, self.num_prototype, self.hid_dim)	# [30, 10, 400]
			#target_user_rep = target_user_rep + neighbor_info
			target_user_rep = neighbor_info


		elif self.gnn_mode == 'mgat2':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			#nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			#nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			#nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)	# [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			#nei2_user_rep = self.user_encoder(nei2_his_rep)
			#target_user_rep, nei1_user_rep, nei2_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1), nei2_user_rep.transpose(0,1)
			target_user_rep, nei1_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1)

			#neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2), nei2_user_rep.unsqueeze(dim = 2)), dim = 2)
			neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2)), dim = 2)
			num_neighbor = neighbor_user_rep.size(2)

			t = torch.matmul(neighbor_user_rep, self.w1) + torch.matmul(target_user_rep.unsqueeze(dim = 2).repeat(1, 1, num_neighbor, 1), self.w2)
			att = torch.tanh(torch.matmul(t, self.w4.unsqueeze(dim = 1).repeat(1, t.size(1), 1, 1)))
			att = F.softmax(att, dim = 2)

			target_user_rep = torch.matmul(att.transpose(-1, -2), neighbor_user_rep).squeeze(dim = 2)
			target_user_rep = target_user_rep.transpose(0, 1)

		elif self.gnn_mode == 'sgat':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)    # [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			nei2_user_rep = self.user_encoder(nei2_his_rep)
			#target_user_rep, nei1_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1)
			target_user_rep, nei1_user_rep, nei2_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1), nei2_user_rep.transpose(0,1)

			#neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2)), dim = 2)
			neighbor_user_rep = torch.cat((target_user_rep.unsqueeze(dim = 2), nei1_user_rep.unsqueeze(dim = 2), nei2_user_rep.unsqueeze(dim = 2)), dim = 2)
			num_neighbor = neighbor_user_rep.size(2)

			t = torch.matmul(neighbor_user_rep, self.w1) + torch.matmul(target_user_rep.unsqueeze(dim = 2).repeat(1, 1, num_neighbor, 1), self.w2)
			att = torch.tanh(torch.matmul(t, self.w5))
			att = torch.matmul(att, self.proj)
			att = F.softmax(att, dim = 2)
			target_user_rep = torch.matmul(att.transpose(-1, -2), neighbor_user_rep).squeeze(dim = 2)
			target_user_rep = target_user_rep.transpose(0, 1)


		elif self.gnn_mode == 'sgcn':
			nei1_his_rep = self.news_encoder(nei1_title, nei1_abstract)
			#nei2_his_rep = self.news_encoder(nei2_title, nei2_abstract)
			nei1_his_rep = self.attention(nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0), nei1_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			#nei2_his_rep = self.attention(nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0), nei2_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)
			nei1_his_rep = self.multi_rep_encoder(nei1_his_rep)
			#nei2_his_rep = self.multi_rep_encoder(nei2_his_rep)	# [30, 50, 10, 400], [30, 50, 10, 400]
			nei1_user_rep = self.user_encoder(nei1_his_rep)
			#nei2_user_rep = self.user_encoder(nei2_his_rep)

			#target_user_rep, nei1_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1)
			#target_user_rep, nei1_user_rep, nei2_user_rep = target_user_rep.transpose(0,1), nei1_user_rep.transpose(0,1), nei2_user_rep.transpose(0,1)
			#target_user_rep = torch.matmul(target_user_rep, self.wgcn1) + torch.matmul(nei1_user_rep, self.wgcn2)
			target_user_rep = torch.matmul(target_user_rep, self.wgcn1) + torch.matmul(nei1_user_rep, self.wgcn2) + torch.matmul(nei2_user_rep, self.wgcn3)

		else:
			pass



		if self.contrastive_mode == 'USER':
			user_infoNCE_logits = self.infoNCE(target_user_rep)
			return predict_logits, user_infoNCE_logits
		else:
			return predict_logits, None


class InfoNCE(nn.Module):
	def __init__(self, hid_dim, infonce_mode, prototype):
		super().__init__()
		self.mode = infonce_mode
		self.prototype = prototype

		self.W = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.W, gain = 1.414)

	def forward(self, multi_rep):
		if self.mode == 'prototype_self':
		# 正负样本1:1，正样本为对应兴趣原型向量，负样本为随机抽取某兴趣条件下新闻语义表示
		# anchor: n_i_k, positive: p_k, negative: n_i_k'
			positive_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1, )).cuda()
			negative_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1, )).cuda()
			while (positive_index == negative_index):
				negative_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1,)).cuda()

			anchor = torch.index_select(multi_rep, dim = 1, index = positive_index)    # [30, 1, 400]
			positive = torch.index_select(self.prototype, dim = 0, index = positive_index)	  # [1, 400]
			negative = torch.index_select(multi_rep, dim = 1, index = negative_index)	# [30, 1, 400]

			positive_logit = torch.matmul(anchor.squeeze(dim = 1), positive.transpose(-1, -2))	   # [30, 1]
			negative_logit = torch.matmul(anchor, negative.transpose(-1, -2)).squeeze(dim = 2)	   # [30, 1]
			logits = torch.cat([positive_logit, negative_logit], dim = -1)
		
		elif self.mode == 'prototype_other':
		# 正负样本1:(n-1)，正样本为对应兴趣原型向量，负样本为其余兴趣条件下新闻语义表示
		# anchor: n_i_k, positive: p_k, negative: [n_i_1, ……, n_i_k', ……, n_i_(nump-1)]
			positive_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1, ))
			negative_index = torch.LongTensor(list(set(list(range(multi_rep.size(1)))) - set(positive_index.numpy().tolist())))
			positive_index, negative_index = positive_index.cuda(), negative_index.cuda()

			anchor = torch.index_select(multi_rep, dim = 1, index = positive_index)    # [30, 1, 400]
			positive = torch.index_select(self.prototype, dim = 0, index = positive_index)	  # [1, 400]
			negative = torch.index_select(multi_rep, dim = 1, index = negative_index)	# [30, 4, 400]

			positive_logit = torch.matmul(anchor.squeeze(dim = 1), positive.transpose(-1, -2))	   # [30, 1]
			negative_logit = torch.matmul(anchor, negative.transpose(-1, -2)).squeeze(dim = 1)	   # [30, 4]
			logits = torch.cat([positive_logit, negative_logit], dim = -1)

		elif self.mode == 'prototype_other_2':
		# 正负样本1:(n-1)，正样本为对应兴趣原型向量，负样本为其余兴趣条件下新闻语义表示
		# anchor: n_i_k, positive: p_k, negative: [n_i_1, ……, n_i_k', ……, n_i_(nump-1)]
			positive_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1, ))
			negative_index = torch.LongTensor(list(set(list(range(multi_rep.size(1)))) - set(positive_index.numpy().tolist())))
			positive_index, negative_index = positive_index.cuda(), negative_index.cuda()

			anchor = torch.index_select(multi_rep, dim = 1, index = positive_index)    # [30, 1, 400]
			positive = torch.index_select(self.prototype, dim = 0, index = positive_index)	  # [1, 400]
			negative = torch.index_select(self.prototype, dim = 0, index = negative_index)	# [4, 400]

			positive_logit = torch.matmul(anchor.squeeze(dim = 1), positive.transpose(-1, -2))	   # [30, 1]
			negative_logit = torch.matmul(anchor.squeeze(dim = 1), negative.transpose(-1, -2))	   # [30, 4]
			logits = torch.cat([positive_logit, negative_logit], dim = -1)

		elif self.mode == 'inter_shuffle_other':
		# 正负样本1:(n-1)，正样本为batch内随机抽取某新闻在相同兴趣条件下的语义表示，负样本为其余兴趣条件下新闻语义表示
		# anchor: n_i_k, positive: n_j_k, negative: [n_i_1, ……, n_i_k', ……, n_i_(nump-1)]
			positive_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1, ))
			negative_index = torch.LongTensor(list(set(list(range(multi_rep.size(1)))) - set(positive_index.numpy().tolist())))
			positive_index, negative_index = positive_index.cuda(), negative_index.cuda()

			shuffle_index = list(range(multi_rep.size(0)))
			random.shuffle(shuffle_index)
			shuffle_index = torch.LongTensor(shuffle_index).cuda()
			multi_rep_shuffle = torch.index_select(multi_rep, dim = 0, index = shuffle_index)

			anchor = torch.index_select(multi_rep, dim = 1, index = positive_index)    # [30, 1, 400]
			positive = torch.index_select(multi_rep_shuffle, dim = 1, index = positive_index)	 # [30, 1, 400]
			negative = torch.index_select(multi_rep, dim = 1, index = negative_index)	# [30, 4, 400]

			positive_logit = torch.matmul(anchor, positive.transpose(-1, -2)).squeeze(dim = 1)	   # [30, 1]
			negative_logit = torch.matmul(anchor, negative.transpose(-1, -2)).squeeze(dim = 1)	   # [30, 4]

			logits = torch.cat([positive_logit, negative_logit], dim = -1)
		
		elif self.mode == 'inter_min_other':
		# 正负样本1:(n-1)，正样本为batch内在相同兴趣条件下与其相似度最低的语义表示，负样本为其余兴趣条件下新闻语义表示
		# anchor: n_i_k, positive: min(n_j_k), negative: [n_i_1, ……, n_i_k', ……, n_i_(nump-1)]
			positive_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1, ))
			negative_index = torch.LongTensor(list(set(list(range(multi_rep.size(1)))) - set(positive_index.numpy().tolist())))
			positive_index, negative_index = positive_index.cuda(), negative_index.cuda()

			anchor = torch.index_select(multi_rep, dim = 1, index = positive_index)    # [30, 1, 400]
			negative = torch.index_select(multi_rep, dim = 1, index = negative_index)	# [30, 4, 400]
			
			positive_logit = torch.matmul(anchor.squeeze(dim=1), anchor.squeeze(dim=1).transpose(-1, -2))	  # [30, 30]
			positive_logit = torch.min(positive_logit, dim = 1).values.reshape(multi_rep.size(0), 1)
			negative_logit = torch.matmul(anchor, negative.transpose(-1, -2)).squeeze(dim = 1)	   # [30, 4]

			logits = torch.cat([positive_logit, negative_logit], dim = -1)
		
		elif self.mode == 'extrapolation':
		# 正负样本1:(n-1)，正样本为该兴趣条件下新闻语义与对应原型的外插值表示，负样本为其余兴趣条件下新闻语义表示
		# anchor: n_i_k, positive: alpha * (n_i_k - p_k) + n_i_k , negative: [n_i_1, ……, n_i_k', ……, n_i_(nump-1)]
			positive_index = torch.randint(low = 0, high = multi_rep.size(1), size = (1, ))
			negative_index = torch.LongTensor(list(set(list(range(multi_rep.size(1)))) - set(positive_index.numpy().tolist())))
			positive_index, negative_index = positive_index.cuda(), negative_index.cuda()

			anchor = torch.index_select(multi_rep, dim = 1, index = positive_index)    # [30, 1, 400]
			positive = torch.index_select(self.prototype, dim = 0, index = positive_index)	  # [1, 400]
			negative = torch.index_select(multi_rep, dim = 1, index = negative_index)	# [30, 4, 400]

			positive_logit = torch.matmul(anchor.squeeze(dim = 1), positive.transpose(-1, -2))	   # [30, 1]
			negative_logit = torch.matmul(anchor, negative.transpose(-1, -2)).squeeze(dim = 1)	   # [30, 4]
			logits = torch.cat([positive_logit, negative_logit], dim = -1)


		return logits



#CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 nohup python -u main.py --num_dataset 6 --num_epoch 3 --num_prototype 5 --batch_size 800 --alpha 1.0 
#--dropout_rate 0.0 --multi_rep_mode concat --infonce_mode prototype_other --contrastive_mode USER --gnn_mode sgat --agg_mode soft -
#-preserve_dir concat_dr0.0_prototype_other_user_sgat_soft_6_3_5_s > concat_dr0.0_prototype_other_user_sgat_soft_6_3_5_s.out &

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u main.py --num_dataset 6 --num_epoch 3 --num_prototype 5 --batch_size 1500 --alpha 1.0 
#--dropout_rate 0.0 --multi_rep_mode concat --infonce_mode prototype_other --contrastive_mode None --gnn_mode nogat --agg_mode soft 
#--preserve_dir concat_dr0.0_prototype_other_none_nogat_soft_6_3_5_l > concat_dr0.0_prototype_other_none_nogat_soft_6_3_5_l.out &