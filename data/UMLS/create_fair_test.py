import torch
import os
import numpy as np


def read_entity_from_id(filename='./entity2id.txt'):
	entity2id = {}
	with open(filename, 'r') as f:
		for line in f:
			if len(line.strip().split()) > 1:
				entity, entity_id = line.strip().split(
				)[0].strip(), line.strip().split()[1].strip()
				entity2id[entity] = int(entity_id)
	return entity2id


def read_relation_from_id(filename='./relation2id.txt'):
	relation2id = {}
	with open(filename, 'r') as f:
		for line in f:
			if len(line.strip().split()) > 1:
				relation, relation_id = line.strip().split(
				)[0].strip(), line.strip().split()[1].strip()
				relation2id[relation] = int(relation_id)

	# relation corresponding to self loop
	# relation2id['self_loop'] = len(relation2id)
	return relation2id

def parse_line(line):
	line = line.strip().split()
	e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
	return e1, relation, e2


def load_data_train(filename, entity2id, relation2id):
	with open(filename) as f:
		lines = f.readlines()

	# this is list for relation triples
	triples_data = []

	# for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
	# columnn of sparse tensor, data contains the type of relation
	# Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
	# type they are connected with
	rows, cols, data = [], [], []
	unique_entities = set()
	for line in lines:
		e1, relation, e2 = parse_line(line)
		unique_entities.add(e1)
		unique_entities.add(e2)
		triples_data.append(
			(entity2id[e1], relation2id[relation], entity2id[e2]))

		# Connecting tail and source entity
		rows.append(entity2id[e2])
		cols.append(entity2id[e1])
		data.append(relation2id[relation])
	# Appending self-loops to adjacency matrix
	# for i in range(len(entity2id)):
	#     rows.append(i)
	#     cols.append(i)
	#     if is_unweigted:
	#         data.append(1)
	#     else:
	#         data.append(relation2id['self_loop'])  # corresponding to self-loop
	# print("number of unique_entities ->", len(unique_entities))
	# print(len(rows))
	return triples_data, (rows, cols, data), list(unique_entities)

def load_data_valid(filename, entity2id, relation2id):
	with open(filename) as f:
		lines = f.readlines()

	# this is list for relation triples
	triples_data = []

	# for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
	# columnn of sparse tensor, data contains the type of relation
	# Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
	# type they are connected with
	rows, cols, data = [], [], []
	unique_entities = set()
	for line in lines:
		e1, relation, e2 = parse_line(line)
		unique_entities.add(e1)
		unique_entities.add(e2)
		triples_data.append(
			(entity2id[e1], relation2id[relation], entity2id[e2]))

		# Connecting tail and source entity
		rows.append(entity2id[e2])
		cols.append(entity2id[e1])
		data.append(relation2id[relation])
	# Appending self-loops to adjacency matrix
	# for i in range(len(entity2id)):
	#     rows.append(i)
	#     cols.append(i)
	#     if is_unweigted:
	#         data.append(1)
	#     else:
	#         data.append(relation2id['self_loop'])  # corresponding to self-loop
	# print("number of unique_entities ->", len(unique_entities))
	# print(len(rows))
	return triples_data, (rows, cols, data), list(unique_entities)


def load_data_test(unique_entities_train, filename, entity2id, relation2id):
	with open(filename) as f:
		lines = f.readlines()

	# print(unique_entities_train.keys())
	
	test_triples = []
	for line in lines:
		e1, relation, e2 = parse_line(line)
		if e1 in unique_entities_train.keys() and e2 in unique_entities_train.keys():
			test_triples.append([entity2id[e1], relation2id[relation], entity2id[e2]])
	# print(test_triples)
	return test_triples



def build_data_train():
	entity2id = read_entity_from_id('./entity2id.txt')
	relation2id = read_relation_from_id('./relation2id.txt')
	
	# print("entity2id: {}".format(entity2id))
	# print(relation2id)
	train_triples, _, unique_entities_train = load_data_train('./train.txt', entity2id, relation2id)
	# print(train_triples)
	# save train_triples then
	with open('train2id.txt', 'w') as f:
		f.write(str(len(train_triples)))
		f.write('\n')
		for i in range(len(train_triples)):
			f.write(str(train_triples[i][0]) + ' ' + str(train_triples[i][2]) + ' ' + str(train_triples[i][1]))
			f.write('\n')

	unique_entities_train = { j: i for i, j in enumerate(unique_entities_train) }
	return unique_entities_train


def build_data_test():
	entity2id = read_entity_from_id('./entity2id.txt')
	relation2id = read_relation_from_id('./relation2id.txt')

	unique_entities_train = build_data_train()
	test_triples = load_data_test(unique_entities_train, './test.txt', entity2id, relation2id)
	valid_triples, _, _ = load_data_valid("./valid.txt", entity2id, relation2id)

	with open('test2id.txt', 'w') as f:
		f.write(str(len(test_triples)))
		f.write('\n')
		for i in range(len(test_triples)):
			f.write(str(test_triples[i][0]) + ' ' + str(test_triples[i][2]) + ' ' + str(test_triples[i][1]))
			f.write('\n')
	
	with open('valid2id.txt', 'w') as f:
		f.write(str(len(valid_triples)))
		f.write('\n')
		for i in range(len(valid_triples)):
			f.write(str(valid_triples[i][0]) + ' ' + str(valid_triples[i][2]) + ' ' + str(valid_triples[i][1]))
			f.write('\n')

build_data_test()
