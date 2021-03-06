import argparse
from os.path import dirname, abspath, join, exists
import os

import torch
from torch.optim import Adadelta, Adam, lr_scheduler
from torch.autograd import Variable
from torch import nn
import numpy as np
import random

from download_dataset import DATASETS
from preprocessors import DATASET_TO_PREPROCESSOR
from transformation_tools import adversarial_paraphrase, scoring_rule
import dictionaries
from dataloaders import TextDataset, TextDataLoader
from trainers import Trainer
from evaluators import Evaluator

from models.CharCNN import CharCNN
from models.WordCNN import WordCNN
from models.VDCNN import VDCNN
from models.QRNN import QRNN

import utils

# Random seed
np.random.seed(0)
torch.manual_seed(0)

# Arguments parser
parser = argparse.ArgumentParser(description="Deep NLP Models for Text Classification")
parser.add_argument('--dataset', type=str, default='MR', choices=DATASETS)
parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--initial_lr', type=float, default=0.01)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--do_train', default=False, action='store_true')
parser.add_argument('--do_eval', default=False, action='store_true')
parser.add_argument('--model_path', type=str, default=None)

subparsers = parser.add_subparsers(help='NLP Model')

## WordCNN
WordCNN_parser = subparsers.add_parser('WordCNN')
# WordCNN_parser.set_defaults(preprocess_level='word')
WordCNN_parser.add_argument('--preprocess_level', type=str, default='word', choices=['word', 'char'])
WordCNN_parser.add_argument('--dictionary', type=str, default='WordDictionary', choices=['WordDictionary', 'AllCharDictionary'])
WordCNN_parser.add_argument('--max_vocab_size', type=int, default=50000)
WordCNN_parser.add_argument('--min_count', type=int, default=None)
WordCNN_parser.add_argument('--start_end_tokens', type=bool, default=False)
group = WordCNN_parser.add_mutually_exclusive_group()
group.add_argument('--vector_size', type=int, default=128, help='Only for rand mode')
group.add_argument('--wordvec_mode', type=str, default='glove', choices=['word2vec', 'glove'])
WordCNN_parser.add_argument('--min_length', type=int, default=5)
WordCNN_parser.add_argument('--max_length', type=int, default=300)
WordCNN_parser.add_argument('--sort_dataset', action='store_true')
WordCNN_parser.add_argument('--mode', type=str, default='rand', choices=['rand', 'static', 'non-static', 'multichannel'])
WordCNN_parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])
WordCNN_parser.add_argument('--epochs', type=int, default=10)
WordCNN_parser.set_defaults(model=WordCNN)

## CharCNN
CharCNN_parser = subparsers.add_parser('CharCNN')
CharCNN_parser.set_defaults(preprocess_level='char')
CharCNN_parser.add_argument('--dictionary', type=str, default='CharCNNDictionary', choices=['CharCNNDictionary', 'VDCNNDictionary', 'AllCharDictionary'])
CharCNN_parser.add_argument('--min_length', type=int, default=1014)
CharCNN_parser.add_argument('--max_length', type=int, default=1014)
CharCNN_parser.add_argument('--sort_dataset', action='store_true')
CharCNN_parser.add_argument('--mode', type=str, default='small')
CharCNN_parser.add_argument('--epochs', type=int, default=10)
CharCNN_parser.set_defaults(model=CharCNN)

## VDCNN
VDCNN_parser = subparsers.add_parser('VDCNN')
VDCNN_parser.set_defaults(preprocess_level='char')
VDCNN_parser.add_argument('--dictionary', type=str, default='VDCNNNDictionary', choices=['CharCNNDictionary', 'VDCNNDictionary', 'AllCharDictionary'])
VDCNN_parser.add_argument('--min_length', type=int, default=1014)
VDCNN_parser.add_argument('--max_length', type=int, default=1014)
VDCNN_parser.add_argument('--epochs', type=int, default=3)
VDCNN_parser.add_argument('--depth', type=int, default=29, choices=[9, 17, 29, 49])
VDCNN_parser.add_argument('--embed_size', type=int, default=16)
VDCNN_parser.add_argument('--optional_shortcut', type=bool, default=True)
VDCNN_parser.add_argument('--k', type=int, default=10)
VDCNN_parser.set_defaults(model=VDCNN)

## QRNN
QRNN_parser = subparsers.add_parser('QRNN')
QRNN_parser.add_argument('--preprocess_level', type=str, default='word', choices=['word', 'char'])
QRNN_parser.add_argument('--dictionary', type=str, default='WordDictionary', choices=['WordDictionary', 'AllCharDictionary'])
QRNN_parser.add_argument('--max_vocab_size', type=int, default=50000) 
QRNN_parser.add_argument('--min_count', type=int, default=None)
QRNN_parser.add_argument('--start_end_tokens', type=bool, default=False)
group = QRNN_parser.add_mutually_exclusive_group()
group.add_argument('--vector_size', type=int, default=128)
group.add_argument('--wordvec_mode', type=str, default=None, choices=['word2vec', 'glove'])
QRNN_parser.add_argument('--min_length', type=int, default=5)
QRNN_parser.add_argument('--max_length', type=int, default=300) 
QRNN_parser.add_argument('--sort_dataset', action='store_true')
QRNN_parser.add_argument('--hidden_size', type=int, default=300)
QRNN_parser.add_argument('--num_layers', type=int, default=4)
QRNN_parser.add_argument('--kernel_size', type=int, default=2)
QRNN_parser.add_argument('--pooling', type=str, default='fo')
QRNN_parser.add_argument('--zoneout', type=float, default=0.5)
QRNN_parser.add_argument('--dropout', type=float, default=0.3)
QRNN_parser.add_argument('--dense', type=bool, default=True)
QRNN_parser.add_argument('--epochs', type=int, default=10)
QRNN_parser.set_defaults(model=QRNN)

args = parser.parse_args()

# Logging
model_name = args.model.__name__
logger = utils.get_logger(model_name)

logger.info('Arguments: {}'.format(args))

logger.info("Preprocessing...")
Preprocessor = DATASET_TO_PREPROCESSOR[args.dataset]
preprocessor = Preprocessor(args.dataset)
# data format ([tok1, tok2, ..., tokn], label)
train_data, val_data, test_data = preprocessor.preprocess(level=args.preprocess_level)
test_data = random.sample(test_data, 200)
print("train size {}, val size {}, test size {}".format(len(train_data), len(val_data), len(test_data)))

logger.info("Building dictionary...")
Dictionary = getattr(dictionaries, args.dictionary)
dictionary = Dictionary(args)
dictionary.build_dictionary(train_data)

logger.info("Making dataset & dataloader...")
if args.do_train:
    train_dataset = TextDataset(train_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
    train_dataloader = TextDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=args.batch_size)
    val_dataset = TextDataset(val_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
    val_dataloader = TextDataLoader(dataset=val_dataset, dictionary=dictionary, batch_size=64)
if args.do_eval:
    test_dataset = TextDataset(test_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
    test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=64)

logger.info("Constructing model...")
model = args.model(n_classes=preprocessor.n_classes, dictionary=dictionary, args=args)
if args.use_gpu:
    model = model.cuda() 

if args.do_train:
    logger.info("Training...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        optimizer = Adam(params=trainable_params, lr=args.initial_lr)
    if args.optimizer == 'Adadelta':
        optimizer = Adadelta(params=trainable_params, lr=args.initial_lr, weight_decay=0.95)
    lr_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.0001)
    criterion = nn.CrossEntropyLoss
    trainer = Trainer(model, train_dataloader, val_dataloader, 
                    criterion=criterion, optimizer=optimizer, 
                    lr_schedule=args.lr_schedule, lr_scheduler=lr_plateau, 
                    use_gpu=args.use_gpu, logger=logger)
    trainer.run(epochs=args.epochs)

if args.do_eval:
    logger.info("Evaluating...")
    if args.do_train:
        logger.info('Best Model: {}'.format(trainer.best_checkpoint_filepath))
        model.load_state_dict(torch.load(trainer.best_checkpoint_filepath)) # load best model
    else:
        logger.info('Best Model: {}'.format(args.model_path))
        model.load_state_dict(torch.load(args.model_path)) # load pretrained model
    evaluator = Evaluator(model, test_dataloader, use_gpu=args.use_gpu, logger=logger)
    test_acc_message = evaluator.evaluate()

    # adversarial evaluation
    successful_perturbations = 0
    failed_perturbations = 0
    sub_rate_list = []
    NE_rate_list = []
    n_corrects = 0

    logger.info("Transformation Evaluating ...")
    dataset = args.dataset
    if args.dataset == 'ag_news':
        dataset = 'agnews'
    if args.dataset == 'yahoo_answers':
        dataset = 'yahoo'
    
    model.eval()
    for index, (tokens, label) in enumerate(test_data):
        vector = [dictionary.indexer(token) for token in tokens]
        input_tensor = Variable(torch.LongTensor(vector).unsqueeze(0)).cuda() # (1 X T)
        outputs = model(input_tensor) # (1 X T)
        ori_entropy = scoring_rule(outputs.detach().cpu())[0] # float
        true_y = np.argmax(outputs.detach().cpu()).item()
        # print("input: ", input_tensor, 'label: ', label, "pred_y: ", true_y)
        adv_doc, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(
            model=model,
            tokens=tokens,
            ori_entropy=ori_entropy,
            dictionary=dictionary,
            true_y=true_y,
            dataset=dataset,
            verbose=True
        )
        if adv_y != true_y:
            successful_perturbations += 1
            print('{}. Successful example crafted.'.format(index))
        else:
            failed_perturbations += 1
            print('{}. Failure.'.format(index))

            text = adv_doc
            sub_rate_list.append(sub_rate)
            NE_rate_list.append(NE_rate)
            # file_2.write(str(index) + str(change_tuple_list) + '\n')

        correct = adv_y == label # ByteTensor
        n_corrects += int(correct) # FloatTensor
        # file_1.write(text + " sub_rate: " + str(sub_rate) + "; NE_rate: " + str(NE_rate) + "\n")
    # end_cpu = time.clock()

    # print('CPU second:', end_cpu - start_cpu)
    mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list) if len(sub_rate_list) else 0
    mean_NE_rate = sum(NE_rate_list) / len(NE_rate_list) if len(NE_rate_list) else 0
    print('mean substitution rate:', mean_sub_rate)
    print('mean NE rate:', mean_NE_rate)
    logger.info(test_acc_message)
    print('new test accuracy:', n_corrects/len(test_data))
