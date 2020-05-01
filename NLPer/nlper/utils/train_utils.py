import matplotlib.pyplot as plt
import os

from rouge import Rouge


def calculate_rouge(hypothesis, reference):
    rouge = Rouge()
    hypothesis = hypothesis.split('<sos>')[1].split('<eos>')[0].strip()
    reference = reference.split('<sos>')[1].split('<eos>')[0].strip()
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores
    except Exception:
        return None


def draw_attention_matrix(attention, original, summary, config=None, epoch=None, batch_id=None):
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
