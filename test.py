

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

from konlpy.tag import Komoran


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




# ===============================================================================================
# Create Model
MODEL_PATH = 'models/'
D_LAYERS = 2
FILTER = 512
N_CLASSES = len(SENTIMENTS)
# COST_WEIGHTS = [100, 100, 100, 50, 100, 10, 30, 10]
COST_WEIGHTS = []
batch_size = 1
PROJECT = 'review_sentiments_N{}'.format(NUM_KEYS)
learning_rate = 1e-5


g = tf.Graph()
with g.as_default():
	model = models.ReviewClassifier(g, MODEL_PATH + '{}.ckpt'.format(PROJECT),
						n_features=NUM_KEYS, n_classes=N_CLASSES, d_layers=D_LAYERS,
						learn_rate=learning_rate, filter=FILTER, drop_out=0.7, cost_wgt=COST_WEIGHTS)




# ===============================================================================================
# Test

DATA_FILE = 'dataset/reviews_total.xlsx'
pd_reviews = pd.read_excel(DATA_FILE, sheet_name='test')



# TEST_TYPES = ['single', 'multi']
TEST_TYPES = ['single']
for TEST_TYPE in TEST_TYPES:
	#
	results = []
	for index, row in pd_reviews.iterrows():
		print(TEST_TYPE, index, row)
		review = row['review_original']
		sents = row['sentiments']
		if str(sents) == 'nan':
			sents = ''
		sents = sents.split(',')
		# ---------------------------------------
		if TEST_TYPE == 'multi':
			if len(sents) <= 1:
				continue
		else:
			if len(sents) > 1:
				continue
		# ---------------------------------------
		review = review.replace('\n', '. ')
		try:
			poses = nlp.pos(review)
		except:
			poses = []
		#
		key_poses = []
		review_hot = np.zeros(NUM_KEYS)
		for pos in poses:
			if pos[1] in KEY_POSES:
				index_cnt = keyword_dic.get(pos[0])
				if index_cnt is None:
					continue
				key_poses.append(pos[0])
				key_index = index_cnt[0]		# index_cnt
				review_hot[key_index] += 1
		#
		logits, preds, logits_origin = model.test([review_hot])
		# --------------------------------------------------------
		pred = np.round(logits[0], 2)
		pred2 = np.round(logits_origin[0], 2)
		# print(pred)
		pred_sents = {}
		for p in range(len(pred)):
			if p == 0:
				pred_sents.update({"없음": pred[p]})
			elif p == 1:
				pred_sents.update({"만족": pred[p]})
			elif p == 2:
				pred_sents.update({"아쉬움": pred[p]})
			elif p == 3:
				pred_sents.update({"불만": pred[p]})
			elif p == 4:
				pred_sents.update({"비싸다": pred[p]})
		# --------------------------------------------------------
		results.append([index, review, sents, pred_sents])
		print('\t', len(results))
	#
	now = datetime.datetime.now()
	now_str = now.strftime('%m%d')
	pd_results = pd.DataFrame(results)
	pd_results.columns = ['no', 'review', 'sentiments', 'predict']
	pd_results.to_excel('result_{}_{}.xlsx'.format(now_str, TEST_TYPE))

