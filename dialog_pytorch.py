from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, tokenize
from sklearn import metrics
from memn2n_pytorch import MemN2NDialog
from itertools import chain
from six.moves import range, reduce
import sys
import numpy as np
import os
import shutil
import timeit

import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse


def get_bool(arg):
	if arg.lower() in ('true', 't', 'yes'):
		return True
	else:
		return False


parser = argparse.ArgumentParser()
parser.add_argument("-learning_rate", type=float, default=0.001, 
					help="learning rate (default=0.001)")
parser.add_argument("-epsilon", type=float, default=1e-8, 
					help="epsilon (default=1e-8)")
parser.add_argument("-max_grad_norm", type=float, default=40.0, 
					help="Clip gradients to this norm")
parser.add_argument("-evaluation_interval", type=int, default=10, 
					help="Evaluate and print results every x epochs")
parser.add_argument("-batch_size", type=int, default=32, 
					help="Batch size for training")
parser.add_argument("-hops", type=int, default=3, 
					help="Number of hops in the Memory Network")
parser.add_argument("-epochs", type=int, default=200, 
					help="Number of epochs to train for")
parser.add_argument("-embedding_size", type=int, default=20, 
					help="Embedding size for embedding matrices")
parser.add_argument("-memory_size", type=int, default=50, 
					help="Maximum size of memory")
parser.add_argument("-task_id", type=int, default=1, 
					help="bAbI task id, 1 <= id <= 6")
parser.add_argument("-random_state", type=int, default=None, 
					help="Random state.")
parser.add_argument("-data_dir", type=str, default="data/dialog-bAbI-tasks/",
					help="Directory containing bAbI tasks")
parser.add_argument("-model_dir", type=str, default="model/",
					help="Directory containing memn2n model checkpoints")
parser.add_argument('-train', type=get_bool, default=True, 
					help='if True, begin to train')
parser.add_argument('-interactive', type=get_bool, default=False, 
					help='if True, interactive')
parser.add_argument('-OOV', type=get_bool, default=False, 
					help='if True, use OOV test set')
args = parser.parse_args()

print("Started Task:", args.task_id)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	# if is_best:
	# 	shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoit(model, optimizer, path_to_model):
	if os.path.isfile(path_to_model):
		print("=> loading checkpoint '{}'".format(path_to_model))
		checkpoint = torch.load(path_to_model)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})"
			  .format(path_to_model, checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(path_to_model))



class chatBot(object):
	def __init__(self, data_dir, model_dir, task_id,
				isInteractive=True,
				OOV=False,
				memory_size=50,
				random_state=None,
				batch_size=32,
				learning_rate=0.001,
				epsilon=1e-8,
				max_grad_norm=40.0,
				evaluation_interval=10,
				hops=3,
				epochs=200,
				embedding_size=20):

		self.data_dir = data_dir
		self.task_id = task_id
		self.model_dir = model_dir
		self.isInteractive = isInteractive
		self.OOV = OOV
		self.memory_size = memory_size
		self.random_state = random_state
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.max_grad_norm = max_grad_norm
		self.evaluation_interval = evaluation_interval
		self.hops = hops
		self.epochs = epochs
		self.embedding_size = embedding_size

		candidates, self.candid2indx = load_candidates(
			self.data_dir, self.task_id)
		self.n_cand = len(candidates)
		print("Candidate Size", self.n_cand)
		self.indx2candid = dict(
			(self.candid2indx[key], key) for key in self.candid2indx)
		# task data
		self.trainData, self.testData, self.valData = load_dialog_task(
			self.data_dir, self.task_id, self.candid2indx, self.OOV)
		data = self.trainData + self.testData + self.valData
		self.build_vocab(data, candidates)
		# self.candidates_vec=vectorize_candidates_sparse(candidates,self.word_idx)
		self.candidates_vec = vectorize_candidates(
			candidates, self.word_idx, self.candidate_sentence_size)
		self.model = MemN2NDialog(self.batch_size, 
									self.vocab_size, 
									self.n_cand, 
									self.sentence_size, 
									self.embedding_size, 
									self.candidates_vec,
									hops=self.hops, 
									max_grad_norm=self.max_grad_norm, 
									task_id=task_id)


	def build_vocab(self, data, candidates):
		vocab = reduce(lambda x, y: x | y, (set(
			list(chain.from_iterable(s)) + q) for s, q, a in data))
		vocab |= reduce(lambda x, y: x | y, (set(candidate)
											 for candidate in candidates))
		vocab = sorted(vocab)
		self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
		max_story_size = max(map(len, (s for s, _, _ in data)))
		mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
		self.sentence_size = max(
			map(len, chain.from_iterable(s for s, _, _ in data)))
		self.candidate_sentence_size = max(map(len, candidates))
		query_size = max(map(len, (q for _, q, _ in data)))
		self.memory_size = min(self.memory_size, max_story_size)
		self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
		self.sentence_size = max(
			query_size, self.sentence_size)  # for the position
		# params
		print("vocab size:", self.vocab_size)
		print("Longest sentence length", self.sentence_size)
		print("Longest candidate sentence length",
			  self.candidate_sentence_size)
		print("Longest story length", max_story_size)
		print("Average story length", mean_story_size)

	def interactive(self):
		context = []
		u = None
		r = None
		nid = 1
		while True:
			line = input('--> ').strip().lower()
			if line == 'exit':
				break
			if line == 'restart':
				context = []
				nid = 1
				print("clear memory")
				continue
			u = tokenize(line)
			data = [(context, u, -1)]
			s, q, a = vectorize_data(
				data, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)

			s = Variable(torch.from_numpy(np.stack(s)))
			q = Variable(torch.from_numpy(np.stack(q)))
			a = Variable(torch.from_numpy(np.stack(a)))

			preds = list(self.model.predict(s, q).data.numpy().tolist())
			r = self.indx2candid[preds[0]]
			print(r)
			r = tokenize(r)
			u.append('$u')
			u.append('#' + str(nid))
			r.append('$r')
			r.append('#' + str(nid))
			context.append(u)
			context.append(r)
			nid += 1

	def train(self):
		trainS, trainQ, trainA = vectorize_data(
			self.trainData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
		valS, valQ, valA = vectorize_data(
			self.valData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
		n_train = len(trainS)
		n_val = len(valS)
		print("Training Size", n_train)
		print("Validation Size", n_val)
		# tf.set_random_seed(self.random_state)
		batches = zip(range(0, n_train - self.batch_size, self.batch_size),
					  range(self.batch_size, n_train, self.batch_size))
		batches = [(start, end) for start, end in batches]
		best_validation_accuracy = 0

		times = []

		for t in range(1, self.epochs + 1):
			np.random.shuffle(batches)
			total_cost = 0.0

			start_time = timeit.default_timer()

			for start, end in batches:
				s = trainS[start:end]
				q = trainQ[start:end]
				a = trainA[start:end]

				s = Variable(torch.from_numpy(np.stack(s)))
				q = Variable(torch.from_numpy(np.stack(q)))
				a = Variable(torch.from_numpy(np.stack(a)))

				cost_t = self.model.batch_fit(s, q, a)
				total_cost += cost_t.data[0]

			end_time = timeit.default_timer()
			times.append(end_time - start_time)

			if t % self.evaluation_interval == 0:
				train_preds = self.batch_predict(trainS, trainQ, n_train)
				val_preds = self.batch_predict(valS, valQ, n_val)
				train_acc = metrics.accuracy_score(
					np.array(train_preds), trainA)
				val_acc = metrics.accuracy_score(val_preds, valA)
				print('-----------------------')
				print('Epoch', t)
				print('Total Cost:', total_cost)
				print('Training Accuracy:', train_acc)
				print('Validation Accuracy:', val_acc)
				print('Average time per epoch: ', np.sum(times)/len(times))
				print('-----------------------')

				if val_acc > best_validation_accuracy:
					best_validation_accuracy = val_acc
					save_checkpoint({
						'epoch': t + 1,
						'state_dict': self.model.state_dict(),
						'optimizer' : self.model.optimizer.state_dict(),
					}, True, filename=self.model_dir+'best_model')

	def test(self):
		load_checkpoit(self.model, self.model.optimizer, self.model_dir + 'best_model')
		
		if self.isInteractive:
			self.interactive()
		else:
			testS, testQ, testA = vectorize_data(
				self.testData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
			n_test = len(testS)
			print("Testing Size", n_test)
			test_preds = self.batch_predict(testS, testQ, n_test)
			test_acc = metrics.accuracy_score(test_preds, testA)
			print("Testing Accuracy:", test_acc)

	def batch_predict(self, S, Q, n):
		preds = []
		for start in range(0, n, self.batch_size):
			end = start + self.batch_size
			s = S[start:end]
			q = Q[start:end]

			s = Variable(torch.from_numpy(np.array(s)))
			q = Variable(torch.from_numpy(np.array(q)))

			pred = self.model.predict(s, q)
			preds += list(pred.data.numpy().tolist())
		return preds



if __name__ == '__main__':
	model_dir = "task" + str(args.task_id) + "_" + args.model_dir
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	chatbot = chatBot(args.data_dir, model_dir, args.task_id, OOV=args.OOV,
					  isInteractive=args.interactive, batch_size=args.batch_size)
	if args.train:
		chatbot.train()
	else:
		chatbot.test()
