import matplotlib.pyplot as plt
import os
import torch

from rouge import Rouge
from typing import List
from typing import Optional


def calculate_rouge(hypothesis: str, reference: str) -> Optional[List[dict]]:
    """
    Calculates Rouge scores which is a set of metrics for evaluation of machine translation or text summarization tasks.
    Rouge stands for Recall-Oriented Understudy for Gisting Evaluation and compares model output with target text.
    For text summarization task we consider two base accuracy measures - recall and precision.
    * Recall - number of overlapping words, divided by number of words in reference summary
    * Precision - number of overlapping words, divided by number of words in model generated summary

    Additionally we consider F1-score of both recall and precision.

    For the best overview of a model performance, we should measure recall, precision and F-score values.

    There are few type of metrices:
    * ROUGE-1
        Measures overlapping unigrams
    * ROUGE-2
        Measures overlapping bigrams
    * ROUGE-L
        Measures longest common subsequence (LCS), takes into account in-sequence matches on sentence level word order.

    :param hypothesis: Model generated text sequence
    :type hypothesis: str
    :param reference: Reference text sequence
    :type reference: str
    :return: List of precision, recall and F1-score for Rouge-1, Rouge-2 and Rouge-L metrics
    :rtype: list
    """
    rouge = Rouge()
    hypothesis = hypothesis.split('<sos>')[1].split('<eos>')[0].strip()
    reference = reference.split('<sos>')[1].split('<eos>')[0].strip()
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores
    except Exception:
        return None


def draw_attention_matrix(
        attention: torch.Tensor,
        original: str,
        summary: str,
        config=None,
        epoch=None,
        batch_id=None,
) -> None:
    """
    Draws plot with heatmap of attention using matplotlib for particular training step.
    If config specified, saves plot in specified location

    :param attention: Matrix with attention values
    :type attention: torch.Tensor
    :param original: Original text to summarize
    :type original: str
    :param summary: Model generated summary text
    :type summary: str
    :param config: Config, if passed then the plot is saved in config-defined location
    :type config: dict, optional
    :param epoch: Current training epoch for naming purpose in plot saving operation
    :type epoch: int, optional
    :param batch_id: Current batch number for naming purpose in plot saving operation
    :type batch_id: int, optional
    """
    labels_original = original.split('<sos>')[1].split('<eos>')[0].strip().split()
    labels_summary = summary.split('<sos>')[1].split('<eos>')[0].strip().split()
    plt.figure(figsize=(20, 10))
    plt.imshow(attention.numpy()[:len(labels_summary), 1:len(labels_original)])
    plt.xticks([i for i in range(len(labels_original)-1)], labels_original, rotation=75)
    plt.yticks([i for i in range(len(labels_summary))], labels_summary)
    plt.draw()
    if config:
        save_path = os.path.join(
            config['model_output_path'], config['model_name'] + f'_attention_matrix_epoch_{epoch}_batch_{batch_id}.png')
        plt.savefig(save_path)
