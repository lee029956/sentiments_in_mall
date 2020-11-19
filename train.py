

import os
import numpy as np
import pandas as pd
import gzip, pickle
import datetime


import tensorflow as tf			# tensorflow1.14~
from konlpy.tag import Komoran
komoran = Komoran()
nlp = komoran


import models
import utils



# Multi GPU환경에서 GPU번호 설정하여 사용할 때
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '9'





# ==========================================================================================
# Make keyword_dic
KEY_POSES = ['MAG', 'VA', 'VX', 'VV', 'XR', 'NNG', 'NNB', 'NNP']
SENTIMENTS = {"없음": 0, "만족": 1, "아쉬움": 2, "불만": 3, "비싸다": 4}
KEYWORD_DIC_FILE = 'keywords.dict'
NUM_KEYS = 10000					# 키워드 전체 숫자(넉넉하게) => one_hot 길이




try:
	keyword_dic = utils.load_data(KEYWORD_DIC_FILE)
except Exception as ex:
	print(str(ex))


keywords = []
for key, item in keyword_dic.items():
	index, cnt = item
	keywords.append([key, cnt])

keywords.sort(key=lambda x: x[1], reverse=True)
keywords

# ==========================================================================================
# Train data : read excel
DATA_FILE = 'dataset/reviews_total.xlsx'
pd_reviews = pd.read_excel(DATA_FILE, sheet_name='train')



# ==========================================================================================
# Make Dataset
RATIO = [0.0, 0.44, 0.1, 0.45, 0.005]	# 데이터 불균형 반영할 경우
APPLY_RATIO = True



train_X = []
train_Y = []
valid_X = []
valid_Y = []
reviews = []
for index, row in pd_reviews.iterrows():
	# ----------------------------------------------------------------
	# 감성없음
	if str(row['sentiments']) == 'nan':
		sent_one_hot = np.zeros(len(SENTIMENTS))
		sent_one_hot[0] = 1
		#
		key_review = row['review_original']
		key_review = key_review.replace('\n', '. ')
		try:
			poses = nlp.pos(key_review)
		except:
			poses = []
		#
		review_hot = np.zeros(NUM_KEYS)
		key_poses = []
		for pos in poses:
			if pos[1] in KEY_POSES:
				index_cnt = keyword_dic.get(pos[0])
				if index_cnt is None:
					continue
				key_index = index_cnt[0]
				review_hot[key_index] += 1
				key_poses.append(pos[0])
		#
		reviews.append([key_poses, review_hot, sent_one_hot])
		if index % 5 != 0:
			if APPLY_RATIO:
				for i in range(int(1 / RATIO[np.argmax(sent_one_hot)])):
					train_X.append(review_hot)
					train_Y.append(sent_one_hot)
			else:
				train_X.append(review_hot)
				train_Y.append(sent_one_hot)
		else:
			if APPLY_RATIO:
				for i in range(int(1 / RATIO[np.argmax(sent_one_hot)])):
					valid_X.append(review_hot)
					valid_Y.append(sent_one_hot)
			else:
				valid_X.append(review_hot)
				valid_Y.append(sent_one_hot)
	#
	# ----------------------------------------------------------------
	# 감성있음
	else:
		for key_i in range(1, len(SENTIMENTS)):
			sent_one_hot = np.zeros(len(SENTIMENTS))
			key_review = row['key_{}'.format(key_i)]
			if str(key_review) == 'nan':
				continue
			sent_one_hot[key_i] = 1
			#
			# ------------------------------------------------------------------------------
			key_review = key_review.replace('\n', '. ')
			try:
				poses = nlp.pos(key_review)
			except:
				poses = []
			#
			review_hot = np.zeros(NUM_KEYS)
			key_poses = []
			for pos in poses:
				if pos[1] in KEY_POSES:
					key_index = keyword_dic.get(pos[0])[0]		# index_cnt
					review_hot[key_index] += 1
					key_poses.append(pos[0])
			#
			reviews.append([key_poses, review_hot, sent_one_hot])
			if index % 5 != 0:
				if APPLY_RATIO:
					for i in range(int(1 / RATIO[np.argmax(sent_one_hot)])):
						train_X.append(review_hot)
						train_Y.append(sent_one_hot)
				else:
					train_X.append(review_hot)
					train_Y.append(sent_one_hot)
			else:
				if APPLY_RATIO:
					for i in range(int(1 / RATIO[np.argmax(sent_one_hot)])):
						valid_X.append(review_hot)
						valid_Y.append(sent_one_hot)
				else:
					valid_X.append(review_hot)
					valid_Y.append(sent_one_hot)



train_X = np.array(train_X)
train_Y = np.array(train_Y)
valid_X = np.array(valid_X)
valid_Y = np.array(valid_Y)
list(np.sum(train_Y, axis=0) / np.sum(train_Y))
list(np.sum(valid_Y, axis=0) / np.sum(valid_Y))



# ===============================================================================================
# Create Model
MODEL_PATH = 'models/'
D_LAYERS = 2
FILTER = 512
N_CLASSES = len(SENTIMENTS)




# ===============================================================================================
# COST_WEIGHTS = [100, 100, 100, 50, 100, 10, 30, 10]
COST_WEIGHTS = []
# learning_rate = 1e-4
learning_rate = 1e-5
PROJECT = 'review_sentiments_N{}'.format(NUM_KEYS)



g = tf.Graph()
with g.as_default():
	model = models.ReviewClassifier(g, MODEL_PATH + '{}.ckpt'.format(PROJECT),
						n_features=NUM_KEYS, n_classes=N_CLASSES, d_layers=D_LAYERS,
						learn_rate=learning_rate, filter=FILTER, drop_out=0.7, cost_wgt=COST_WEIGHTS)



# ==================================================================================================
# Training
START_ACC = 0.01
START_COST = 1000
best_acc = START_ACC
best_cost = START_COST



if True:
	batch_size = 1000
	for e in range(300):
		for phase in ['train', 'val']:
			if phase == 'train':
				IS_TRAIN = True
				features = train_X
				labels = train_Y
			else:
				IS_TRAIN = False
				features = valid_X
				labels = valid_Y
			# ---------------------------------------------
			# Training / Evaluation
			cost_sum = 0
			accuracy_sum = 0
			total_cnt = 0
			#
			BATCH_CNT = int(len(features) / batch_size)
			if len(features) % batch_size != 0:
				BATCH_CNT += 1
			#
			for b in range(BATCH_CNT):
				batch_X = features[batch_size * b: batch_size * (b + 1)]
				batch_Y = labels[batch_size * b: batch_size * (b + 1)]
				#
				c, l, p, a = model.train(batch_X, batch_Y, IS_TRAIN)
				cost_sum += c
				accuracy_sum += a
				total_cnt += 1
			#
			# ----------------------------------------------------------------------
			# Statistics
			if total_cnt == 0:
				continue
			epoch_cost = cost_sum / total_cnt
			epoch_acc = accuracy_sum / total_cnt
			if e % 10 == 0:
				print()
				print(e)
				print(phase, ', cost={0:0.4f}'.format(epoch_cost), ', accuracy={0:0.4f}'.format(epoch_acc))
				if phase == 'val':
					print('Best val Acc: {:4f}'.format(best_acc))
					print('Best val Cost: {:4f}'.format(best_cost))
		#
		# ----------------------------------------------------------------------
		# Save Model
		if phase == 'val' and epoch_cost < best_cost:
			best_acc = epoch_acc
			best_cost = epoch_cost
			print('change best_model: {0:0.4f}'.format(epoch_cost))
			model.save(model.ModelName)



