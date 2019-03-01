import json
import os
import nltk
import torch
import csv
import string

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def format_table_string(table_reader):
    column_vals = ''
    tmp_str = ''
    for i, row in enumerate(table_reader):
        if i == 0:
            column_vals = row
        row_val = ''
        for j, val in enumerate(row):
            if j == 0:
                tmp_str += column_vals[j] + ' ' + val
                row_val = val
                continue

            #tmp_str += ' ' + column_vals[j] + ' ' + row_val + ' ' + val
            tmp_str += ' ' + column_vals[j] + ' ' + val
            #tmp_str += val + ' '

        tmp_str += '.\n'

    return tmp_str


class SQuAD():
    def __init__(self, args):
        path = '.data/wikitable/data'
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'
        finetune_examples_path = dataset_path + 'finetune_examples.pt'

        print("preprocessing data files...")
        if not os.path.exists(f'{path}/{args.train_file}l'):
            self.preprocess_file(f'{path}/{args.train_file}')
        if not os.path.exists(f'{path}/{args.dev_file}l'):
            self.preprocess_file(f'{path}/{args.dev_file}')
        if not os.path.exists(f'{path}/{args.finetune_file}l'):
            self.preprocess_our_file(f'{path}/{args.finetune_file}', True)

        self.RAW = data.RawField()
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)
            finetune_examples = torch.load(finetune_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
            self.finetune = data.Dataset(examples=finetune_examples, fields=list_fields)
        else:
            print("building splits...")
            self.train, self.finetune, self.dev = data.TabularDataset.splits(
                path=path,
                train=f'{args.train_file}l',
                validation=f'{args.finetune_file}l',
                test=f'{args.dev_file}l',
                format='json',
                fields=dict_fields)
            #self.finetune = data.TabularDataset.splits(
            #        path=path,
            #        train=f'{args.finetune_file}l',
            #        format='json',
            #        fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)
            torch.save(self.finetune.examples, finetune_examples_path)

        # cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.finetune, self.dev)
        self.WORD.build_vocab(self.train, self.finetune, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))

        print("building iterators...")
        self.train_iter, self.finetune_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.finetune, self.dev),
                                       batch_sizes=[args.train_batch_size, args.train_batch_size, args.dev_batch_size],
                                       device=args.gpu,
                                       sort_key=lambda x: len(x.c_word))

    def preprocess_our_file(self, path, finetune):
        dump = []
        def_wrong = 0

        with open(path, "r", encoding='utf-8') as reader:
            source = json.load(reader)

            for obj in source:
                ex_id = obj['id']
                if finetune and not ex_id.startswith('table'):
                    continue

                table = obj['table']
                if table is None:
                    def_wrong += 1
                    continue

                table_list = table.split('\n')
                table_reader = csv.reader(table_list)
                try:
                    paragraph_text = 'yes no\n' + format_table_string(table_reader)
                except Exception:
                    continue

                doc_tokens = word_tokenize(paragraph_text)

                question_text = obj['question']

                if finetune:
                    orig_answer_text = obj['answer'].lower().strip()
                    start_position = None
                    for i, token in enumerate(doc_tokens):
                        candidate = token
                        for c in string.punctuation:
                            candidate = candidate.replace(c, '')

                        if orig_answer_text == candidate:
                            start_position = i
                    if start_position is None:
                        continue
                    else:
                        end_position = start_position

                else:
                    start_position = 1
                    end_position = 2
                    orig_answer_text = obj['answer']
                
                if not question_text.strip() or not orig_answer_text.strip() or not table.strip():
                    if not table.strip():
                        def_wrong += 1
                    continue

                dump.append(dict([('id', ex_id),
                                  ('context', paragraph_text),
                                  ('question', question_text),
                                  ('answer', orig_answer_text),
                                  ('s_idx', start_position),
                                  ('e_idx', end_position)]))

        print('Number of examples that should be counted as wrong: ' + str(def_wrong))
        print('Total number of valid examples: ' + str(len(dump)))
        with open(f'{path}l', 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)

    def preprocess_file(self, path):
        dump = []

        with open(path, "r", encoding='utf-8') as reader:
            source = csv.reader(reader, delimiter='\t')

            for count, row in enumerate(source):
                if count == 0:
                    continue
                ex_id = row[0]
                question_text = row[1]
                answers = row[3]
                table_file = row[2]

                with open('.data/wikitable/' + table_file, 'r') as table_csv:
                    table_reader = csv.reader(table_csv, delimiter=',', quotechar='"')

                    try:
                        paragraph_text = 'yes no\n' + format_table_string(table_reader)
                    except Exception:
                        continue
                    doc_tokens = word_tokenize(paragraph_text)

                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    for answer in answers.split('|'):
                        orig_answer_text = answer
                        if len(word_tokenize(orig_answer_text)) > 1:
                            continue
                        start_position = None
                        for i, token in enumerate(doc_tokens):
                            candidate = token
                            for c in string.punctuation:
                                candidate = candidate.replace(c, '')

                            if orig_answer_text == candidate:
                                start_position = i
                        if start_position is None:
                            continue
                        else:
                            end_position = start_position
                            break

                    if start_position is None:
                        continue

                    dump.append(dict([('id', ex_id),
                                      ('context', paragraph_text),
                                      ('question', question_text),
                                      ('answer', orig_answer_text),
                                      ('s_idx', start_position),
                                      ('e_idx', end_position)]))

        with open(f'{path}l', 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)
