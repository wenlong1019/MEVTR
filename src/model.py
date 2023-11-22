import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from pixel import PIXELConfig, PIXELModel


def prefix_alignment(textual_output, visual_output, seq_lengths, textual_word_starts, visual_word_starts):
    textual_output_alignment = torch.zeros([len(seq_lengths), max(seq_lengths), textual_output.shape[2]]).to(
        textual_output.device)
    visual_output_alignment = torch.zeros([len(seq_lengths), max(seq_lengths), visual_output.shape[2]]).to(
        visual_output.device)
    for b in range(len(seq_lengths)):
        # textual
        for index, word_ind in enumerate(textual_word_starts[b]):
            if index < max(seq_lengths):
                textual_output_alignment[b, index, :] = textual_output[b, word_ind, :]
        # visual
        for index, word_ind in enumerate(visual_word_starts[b]):
            if index < max(seq_lengths):
                visual_output_alignment[b, index, :] = visual_output[b, word_ind, :]
    return textual_output_alignment, visual_output_alignment


def similarity_constraint(textual_output, visual_output):
    visual_output = F.log_softmax(visual_output, dim=-1)
    textual_output = F.softmax(textual_output, dim=-1)
    similarity = F.kl_div(visual_output, textual_output, reduction='sum')
    kl = 10000 / similarity
    return kl


class Multihead_bilinear_model(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.head_num = settings.head_num
        self.dim_visual_encoder = settings.dim_visual_encoder
        self.U_matrix = create_parameter(settings.head_num, settings.dim_visual_encoder, settings.dim_visual_encoder)
        self.U_bias = create_parameter(settings.dim_visual_encoder, settings.dim_visual_encoder)

    def forward(self, textual_output, visual_output):
        textual_inf = textual_output[:, 0, :]
        visual_inf = visual_output[:, 0, :]
        n_batch = textual_inf.shape[0]

        bilinear_scores = torch.einsum("bi,bj->bij", (visual_inf, textual_inf))
        bilinear_scores = bilinear_scores.reshape(n_batch, self.head_num, -1, self.dim_visual_encoder)
        scores = torch.einsum("bijk,ikm->bijm", (bilinear_scores, self.U_matrix))
        scores = scores.reshape(n_batch, self.dim_visual_encoder, self.dim_visual_encoder).permute(0, 2, 1)
        scores = scores + self.U_bias

        scores = F.softmax(scores, dim=-1)
        context = torch.einsum("bij,bjl->bil", (visual_output, scores))
        return context


def create_parameter(*size):
    out = torch.nn.Parameter(torch.empty(*size, dtype=torch.float))
    if len(size) > 1:
        torch.nn.init.xavier_uniform_(out)
    else:
        torch.nn.init.uniform_(out)
    return out


class Rep_projector(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.projection1 = nn.Linear(settings.dim_cat, settings.dim_out)
        self.projection2 = nn.Linear(settings.dim_out, settings.dim_out)
        self.relu = nn.ReLU()

    def forward(self, joint_rep):
        joint_rep = self.projection1(joint_rep)
        joint_rep = self.relu(joint_rep)
        joint_rep = self.projection2(joint_rep)
        joint_rep = self.relu(joint_rep)
        return joint_rep


class MEVTR(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.n_labels = len(settings.target_label_switch)
        # visual_encoder
        self.visual_config = PIXELConfig.from_pretrained(settings.visual_model_name_or_path)
        self.visual_encoder = PIXELModel.from_pretrained(settings.visual_model_name_or_path, config=self.visual_config)
        # textual_encoder
        self.textual_encoder = AutoModel.from_pretrained(settings.textual_model_name_or_path)
        # multi—head bilinear model
        self.multihead_bilinear_model = Multihead_bilinear_model(settings)
        #  representation projector
        self.rep_projector = Rep_projector(settings)

        self.projection = nn.Linear(settings.dim_visual_encoder, settings.dim_visual_encoder)
        self.relu = nn.ReLU()
        self.visual_ln = nn.LayerNorm(settings.dim_visual_encoder)
        self.textual_ln = nn.LayerNorm(settings.dim_textual_encoder)
        self.out_fnn = nn.Linear(settings.dim_out, self.n_labels)

    def forward(self, seq_lengths, visual_values, visual_attention_mask, visual_word_starts,
                textual_input_ids, textual_attention_mask, textual_word_starts, run_test):

        textual_encoder_output = \
            self.textual_encoder(input_ids=textual_input_ids, attention_mask=textual_attention_mask)[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        visual_encoder_output = self.visual_encoder(pixel_values=visual_values, attention_mask=visual_attention_mask)[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # prefix alignment
        textual_output, visual_output = prefix_alignment(textual_encoder_output, visual_encoder_output,
                                                         seq_lengths, textual_word_starts, visual_word_starts)

        # multi-head bilinear model
        textual_output_ln = self.textual_ln(textual_output)
        visual_output_ln = self.visual_ln(visual_output)
        visual_output_new = self.multihead_bilinear_model(textual_output_ln, visual_output_ln)

        #  representation projector
        joint_rep = torch.cat((textual_output_ln, visual_output_new), 2)
        high_dimensional_rep = self.rep_projector(joint_rep)

        last_scores = self.out_fnn(high_dimensional_rep)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not run_test:
            # similarity constraint
            visual_output = self.projection(visual_output)
            visual_output = self.relu(visual_output)
            loss_sc = similarity_constraint(textual_output, visual_output)

            return last_scores, loss_sc
        else:
            return last_scores
