""" Customized version of the official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import csv


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset_file, predictions):
    pairs = []
    with open(dataset_file) as dataset_tmp:
        source = json.load(dataset_tmp)

        f1 = exact_match = total = 0

        for obj in source:
            ex_id = obj['id']
            if ex_id not in predictions:
                message = 'Unanswered question ' + ex_id + \
                            ' will receive score 0.'
                continue
            total += 1
            ground_truths = [obj['answer']]
            prediction = predictions[ex_id]
            old_em = exact_match
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)
            if exact_match == old_em:
                pairs.append((obj['answer'], prediction))
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

    with open('manual_dump_file', 'w+') as dump_file:
        for pair in pairs:
            dump_file.write(pair[0] + '\n')
            dump_file.write(pair[1] + '\n\n')

    return {'exact_match': exact_match, 'f1': f1}


def main(args):
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    results = evaluate(args.dataset_file, predictions)
    #print(json.dumps(results))
    return results


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    results = main(args)
    print(results)
