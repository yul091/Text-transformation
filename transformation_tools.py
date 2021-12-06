import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import re
import spacy
from spacy.tokenizer import Tokenizer
from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc, average_precision_score
from paraphrase import PWWS, _compile_perturbed_tokens

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match) # force to only tokenize by space

def common_get_entropy(pos : torch.Tensor):
    pred_prob = F.softmax(pos, dim=-1) # (N, k)
    etp = entropy(pred_prob, axis=-1)/np.log(pos.size(-1)) # normalized to interval [0,1]
    return etp

def common_get_maxpos(pos : torch.Tensor):
    test_pred_pos, _ = torch.max(F.softmax(pos, dim=1), dim=1)
    return 1 - test_pred_pos.detach().cpu().numpy()

def scoring_rule(pos : torch.Tensor, rule='entropy'):
    if rule == 'vanilla':
        return common_get_maxpos(pos)
    else:
        return common_get_entropy(pos)

def evaluate_word_saliency(model, tokens, dictionary):
    """
    tokens: [tok1, tok2, ..., tokn].
    dictionary: objection for token2index.
    """
    word_saliency_list = []
    texts = ' '.join(tokens)
    doc = nlp(texts)
    UNK_IDX = dictionary.indexer('<UNK>')
    inputs = [dictionary.indexer(token) for token in tokens]
    input_tensor = Variable(torch.LongTensor(inputs).unsqueeze(0)).cuda() # (1 X T)
    outputs = model(input_tensor) # (1 X V)
    original_entropy = scoring_rule(outputs.detach().cpu())[0] # float

    for position in range(len(inputs)):
        # get x_i^(\hat)
        wo_word_input = copy.deepcopy(input_tensor) # (1 X T)
        wo_word_input[0][position] = UNK_IDX
        wo_outputs = model(wo_word_input) # (1 X T)
        wo_entropy = scoring_rule(wo_outputs.detach().cpu())[0] # float

        # calculate S(x,w_i) defined in Eq.(6)
        word_saliency = original_entropy - wo_entropy
        word_saliency_list.append((position, doc[position], word_saliency, doc[position].tag_))

    position_word_list = []
    for word in word_saliency_list:
        position_word_list.append((word[0], word[1]))

    return position_word_list, word_saliency_list, doc



def adversarial_paraphrase(model, tokens, ori_entropy, dictionary, true_y, dataset, verbose=True):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text: generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''

    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        '''
        perturbed_vector = None
        perturbed_vector = [dictionary.indexer(token) for token in perturbed_text.split()]
        perturbed_tensor = Variable(torch.LongTensor(perturbed_vector).unsqueeze(0)).cuda() # (1 X T)
        perturbed_outputs = model(perturbed_tensor) # (1 X T)
        # _, perturbed_y = perturbed_outputs.max(dim=1)
        perturbed_entropy = scoring_rule(perturbed_outputs.detach().cpu())[0] # float

        if ori_entropy < 0.3 or perturbed_entropy < ori_entropy - 0.3:
        # if perturbed_entropy < 0.3:
            return True
        else:
            return False

        # if perturbed_y != true_y: 
        #     return True
        # else:
        #     return False

    def heuristic_fn(text, candidate):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        doc = nlp(text)
        origin_vector = None
        perturbed_vector = None
        origin_vector = [dictionary.indexer(token) for token in tokens]
        input_tensor = Variable(torch.LongTensor(origin_vector).unsqueeze(0)).cuda() # (1 X T)
        perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
        perturbed_doc = nlp(' '.join(perturbed_tokens))
        perturbed_vector = [dictionary.indexer(token) for token in perturbed_doc.text.split()]
        perturbed_tensor = Variable(torch.LongTensor(perturbed_vector).unsqueeze(0)).cuda() # (1 X T)

        origin_outputs = model(input_tensor) # (1 X T)
        origin_entropy = scoring_rule(origin_outputs.detach().cpu())[0] # float
        perturbed_outputs = model(perturbed_tensor) # (1 X T)
        perturbed_entropy = scoring_rule(perturbed_outputs.detach().cpu())[0] # float
        # origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
        # perturbed_prob = grad_guide.predict_prob(input_vector=perturbed_vector)
        delta_p = origin_entropy - perturbed_entropy

        return delta_p

    # PWWS
    position_word_list, word_saliency_list, doc = evaluate_word_saliency(model, tokens, dictionary)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(doc,
                                                                true_y,
                                                                dataset,
                                                                word_saliency_list=word_saliency_list,
                                                                heuristic_fn=heuristic_fn,
                                                                halt_condition_fn=halt_condition_fn,
                                                                verbose=verbose)

    # print("perturbed_text after perturb_text:", perturbed_text)
    origin_vector = perturbed_vector = None
    origin_vector = [dictionary.indexer(token) for token in tokens]
    origin_tensor = Variable(torch.LongTensor(origin_vector).unsqueeze(0)).cuda() # (1 X T)
    origin_outputs = model(origin_tensor) # (1 X T)
    origin_entropy = scoring_rule(origin_outputs.detach().cpu())[0] # float
    # origin_vector = text_to_vector(input_text, tokenizer, dataset)
    perturbed_vector = [dictionary.indexer(token) for token in perturbed_text.split()]
    perturbed_tensor = Variable(torch.LongTensor(perturbed_vector).unsqueeze(0)).cuda() # (1 X T)
    perturbed_outputs = model(perturbed_tensor) # (1 X T)
    _, perturbed_y = perturbed_outputs.max(dim=1)
    perturbed_entropy = scoring_rule(perturbed_outputs.detach().cpu())[0] # float
    # perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
    # perturbed_y = grad_guide.predict_classes(input_vector=perturbed_vector)
    if verbose:
        raw_score = origin_entropy - perturbed_entropy
        # print('Pred before: ', origin_outputs, 'Pred after: ', perturbed_outputs)
        print('Entropy before: ', origin_entropy, '. Entropy after: ', perturbed_entropy,
              '. Entropy shift: ', raw_score)
    return perturbed_text, perturbed_y.item(), sub_rate, NE_rate, change_tuple_list