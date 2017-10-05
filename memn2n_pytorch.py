import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import math



class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, window_size, embedding_size, sentence_size, num_features=10):
		super(Encoder, self).__init__()
		
		self.window_size = window_size
		self.embedding_size = embedding_size
		self.sentence_size = sentence_size
		self.num_features = num_features
		self.output_H = sentence_size

		self.conv1 = nn.Conv2d(1, self.num_features, (self.window_size, self.embedding_size))
		self.output_H = (self.output_H - self.window_size) + 1
		self.conv2 = nn.Conv2d(self.num_features, self.num_features, (self.window_size, 1))
		self.output_H = (self.output_H - self.window_size) + 1

		self.linear = nn.Linear(self.output_H * self.num_features, self.embedding_size)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.Linear):
				n = self.output_H * self.num_features
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()



	def forward(self, sentence):

		out = sentence.contiguous().view(-1, 1, self.sentence_size, self.embedding_size)
		out = nn.functional.relu(self.conv1(out))
		out = nn.functional.relu(self.conv2(out))
		out = self.linear(out.view(-1, self.output_H * self.num_features))
		
		return out




class MemN2NDialog(nn.Module):
	"""docstring for MemN2NDialog"""
	def __init__(self, batch_size, vocab_size, candidates_size, sentence_size, embedding_size,
				candidates_vec,
				candidates_mask,
				hops=3,
				max_grad_norm=40.0,
				nonlin=None,
				optimizer=optim.Adam,
				name='MemN2NDialog',
				task_id=1):
		super(MemN2NDialog, self).__init__()

		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.candidates_size = candidates_size
		self.sentence_size = sentence_size
		self.embedding_size = embedding_size
		self.hops = hops
		self.max_grad_norm = max_grad_norm
		self.nonlin = nonlin
		self.name = name
		self.candidates = candidates_vec
		self.candidates_mask = candidates_mask

		self.embed_A = nn.Embedding(self.vocab_size, self.embedding_size) 
		self.linear_H = nn.Linear(self.embedding_size * 2, self.embedding_size * 2)
		self.embed_W = nn.Embedding(self.vocab_size, self.embedding_size)

		self.softmax = nn.Softmax()
		self.cross_entropy_loss = nn.CrossEntropyLoss()

		# self.encoder = Encoder(3, embedding_size * 2, sentence_size)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
				m.weight.data.normal_(0, 0.1)
			if isinstance(m, nn.Embedding):
				m.weight.data[0].zero_()

		self.optimizer = optimizer(self.parameters(), lr=1e-2)



	def forward(self, stories, query, stories_mask, query_mask):
		return self.inference(stories, query, stories_mask, query_mask)



	def inference(self, stories, query, stories_mask, query_mask):

		# embed query
		query_emb = self.embed_A(query)

		# embed query mask
		query_mask_emb = self.embed_A(query_mask)

		# aplly query mask (add_)
		# query_emb.add_(query_mask_emb)
		query_emb = torch.cat([query_emb, query_mask_emb], 2)
		query_emb_sum = torch.sum(query_emb, 1)
		# query_emb_sum = self.encoder(query_emb)
		u = [query_emb_sum]

		for _ in range(self.hops):

			# embed stories
			stories_unbound = torch.unbind(stories, 1)
			embed_stories = [self.embed_A(story) for story in stories_unbound]
			embed_stories = torch.stack(embed_stories, 1)

			# embed stories mask
			stories_mask_unbound = torch.unbind(stories_mask, 1)
			embed_stories_mask = [self.embed_A(story) for story in stories_mask_unbound]
			embed_stories_mask = torch.stack(embed_stories_mask, 1)

			# aplly stories mask (add_)
			# embed_stories.add_(embed_stories_mask)
			embed_stories = torch.cat([embed_stories, embed_stories_mask], 3)
			embed_stories_sum = torch.sum(embed_stories, 2)
			# embed_stories_sum = torch.unbind(embed_stories, 1)
			# embed_stories_sum = [self.encoder(story) for story in embed_stories_sum]
			# embed_stories_sum = torch.stack(embed_stories_sum, 1)

			# get attention
			u_temp = torch.transpose(torch.unsqueeze(u[-1], -1), 1, 2)
			attention = torch.sum(embed_stories_sum * u_temp, 2)
			attention = self.softmax(attention)

			attention = torch.unsqueeze(attention, -1)
			attn_stories = torch.sum(attention*embed_stories_sum, 1)

			# output = self.linear_H(torch.cat([u[-1], attn_stories], 1))
			new_u = self.linear_H(u[-1]) + attn_stories

			u.append(new_u)

		# embed candidates
		candidates_emb = self.embed_W(self.candidates)

		# embed candidates mask
		candidates_mask_emb = self.embed_W(self.candidates_mask)

		# apply mask (add_)
		# candidates_emb.add_(candidates_mask_emb)
		candidates_emb = torch.cat([candidates_emb, candidates_mask_emb], 2)
		candidates_emb_sum = torch.sum(candidates_emb, 1)
		# candidates_emb_sum = self.encoder(candidates_emb)
		output = torch.mm(new_u, torch.transpose(candidates_emb_sum, 0, 1))

		return output



	def batch_fit(self, stories, query, answers, stories_mask, query_mask):
		self.train()
		# calculate loss
		logits = self.forward(stories, query, stories_mask, query_mask)
		cross_entropy = self.cross_entropy_loss(logits, answers)
		loss = torch.sum(cross_entropy)
		
		self.optimize(loss)
		return loss

	def optimize(self, loss):
		# calculate and apply grads 
		self.optimizer.zero_grad()
		loss.backward()

		nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Embedding):
				m.weight.grad.data[0].zero_()

		self.optimizer.step()


	def predict(self, stories, query, stories_mask, query_mask):
		self.eval()
		# calculate loss
		logits = self.forward(stories, query, stories_mask, query_mask)
		_, preds = torch.max(logits, 1)
		
		return preds


	# def _train(self, stories, query, answers):
	# 	self.train()

	# 	total_cost = 0.0
	# 	batches = zip(range(0, n_train - self.batch_size, self.batch_size),
	# 				range(self.batch_size, n_train, self.batch_size))
	# 	batches = [(start, end) for start, end in batches]
	# 	# run epoch
	# 	for start, end in batches:
	# 		s = trainS[start:end]
	# 		q = trainQ[start:end]
	# 		a = trainA[start:end]

	# 		# calculate loss
	# 		logits = self.forward(s, q)
	# 		cross_entropy = nn.cross_entropy_loss(logits, a)
	# 		loss = torch.sum(cross_entropy)
	# 		total_cost += loss

	# 		# calculate and apply grads 
	# 		self.optimizer.zero_grad()
	# 		loss.backward()
	# 		self.optimizer.step()

	# 	return total_cost



	# def _eval(self, stories, query, answers):
	# 	self.eval()

	# 	preds = self._predict(stories, query)
	# 	val_acc = metrics.accuracy_score(preds, answers)

	# 	return val_acc



	# def _predict(self, stories, query):

	# 	output = self.forward(stories, query)
	# 	preds = nn.functional.softmax(output)
	# 	_, preds = torch.max(preds, 1)

	# 	return preds
