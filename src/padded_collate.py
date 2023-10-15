import torch


class PaddedBatch:
    """Container class for padded data"""

    def __init__(self, graph_ids, seq_lengths, targets,
                 visual_values, visual_attention_mask, visual_word_starts,
                 textual_input_ids, textual_attention_mask, textual_word_starts):
        self.graph_ids = graph_ids
        self.seq_lengths = seq_lengths
        self.targets = targets

        self.visual_values = visual_values
        self.visual_attention_mask = visual_attention_mask
        self.visual_word_starts = visual_word_starts

        self.textual_input_ids = textual_input_ids
        self.textual_attention_mask = textual_attention_mask
        self.textual_word_starts = textual_word_starts

    def to(self, device):
        self.visual_values = torch.stack(self.visual_values).to(device)
        self.visual_attention_mask = torch.stack(self.visual_attention_mask).to(device)
        self.textual_input_ids = torch.tensor(self.textual_input_ids).to(device)
        self.textual_attention_mask = torch.tensor(self.textual_attention_mask).to(device)


def padded_collate(batch):
    # Sort batch by the longest sequence desc
    batch.sort(key=lambda sequence: len(sequence[4]), reverse=True)
    graph_ids, targets, \
        visual_values, visual_attention_mask, visual_word_starts, \
        textual_input_ids, textual_attention_mask, textual_word_starts = zip(*batch)
    seq_lengths = torch.LongTensor([len(indices) for indices in visual_word_starts])
    padded = PaddedBatch(graph_ids, seq_lengths, targets,
                         visual_values, visual_attention_mask, visual_word_starts,
                         textual_input_ids, textual_attention_mask, textual_word_starts)

    return padded
