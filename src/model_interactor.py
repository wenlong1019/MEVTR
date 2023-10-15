import json
import math

import torch
import torch.nn.functional as F
from seqeval.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MEVTR
from padded_collate import padded_collate
from pixel import resize_model_embeddings
from preprocessing import MEVTR_Dataset


class ModelInteractor:
    def __init__(self, settings):
        self.train_data = None
        self.test_data = None
        self.optimizer = None
        self.scheduler = None

        self.settings = settings
        self.device = settings.device
        self.batch_size = settings.batch_size
        self.step = settings.step

        self.model = MEVTR(settings)
        self.model = self.model.to(self.settings.device)

        if self.settings.freeze_layer is not None:
            self.freeze_params(self.settings.freeze_layer)

        self._init_optimizer()
        self._store_settings()

    def _init_optimizer(self):
        textual_params = []
        visual_params = []
        other_params = []

        for name, para in self.model.named_parameters():
            if para.requires_grad:
                if "bert" in name:
                    textual_params += [para]
                elif "pixel" in name:
                    visual_params += [para]
                else:
                    other_params += [para]
        params = [
            {"params": textual_params, "lr": self.settings.lr_textual_encoder},
            {"params": visual_params, "lr": self.settings.lr_visual_encoder},
            {"params": other_params, "lr": self.settings.lr_other}
        ]

        self.optimizer = torch.optim.AdamW(
            params,
            betas=(self.settings.beta1, self.settings.beta2),
            eps=1e-08,
            weight_decay=self.settings.l2)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

    def _store_settings(self):
        with open(self.settings.dir + "settings.json", "w") as fh:
            json.dump({k: v for k, v in self.settings.__dict__.items() if k not in "device".split()}, fh)

    def freeze_params(self, freeze_layer):
        for i in list(range(0, freeze_layer)):
            for param in self.model.textual_encoder.encoder.layer[i].parameters():
                param.requires_grad = False
            for param in self.model.visual_encoder.encoder.layer[i].parameters():
                param.requires_grad = False
        # for name, parameters in self.model.named_parameters():
        #     if parameters.requires_grad:
        #         print(name)

    def _init_training_data(self, train_path):
        self.train_data = MEVTR_Dataset(
            train_path,
            visual_encoder=self.model.visual_encoder,
            textual_encoder=self.model.textual_encoder,
            settings=self.settings)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=padded_collate)

    def _init_test_data(self, test_path):
        self.test_data = MEVTR_Dataset(
            test_path,
            visual_encoder=self.model.visual_encoder,
            textual_encoder=self.model.textual_encoder,
            settings=self.settings)
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=padded_collate)

    def _run_train_batch(self, batch, optimizer):
        optimizer.zero_grad()

        # label [batch x label x head_seq x dependent_seq]
        label_scores, loss_sc = self.model(batch.seq_lengths,
                                           batch.visual_values,
                                           batch.visual_attention_mask,
                                           batch.visual_word_starts,
                                           batch.textual_input_ids,
                                           batch.textual_attention_mask,
                                           batch.textual_word_starts,
                                           run_test=False)
        ###########################################################################
        # loss_ce
        mask_matrix = torch.zeros([len(batch.seq_lengths), batch.seq_lengths[0]]).bool()
        for b in range(len(batch.seq_lengths)):
            mask_matrix[b, 1:batch.seq_lengths[b]] = True
        scores = label_scores[mask_matrix]
        targets = list(batch.targets)
        for b in range(len(batch.seq_lengths)):
            length = batch.seq_lengths[b]
            targets[b] = targets[b][:length - 1]
        gold_targets = []
        for y in targets:
            gold_targets.append(y.flatten())
        gold_targets = torch.cat(gold_targets).cuda().detach().long()
        ###################
        loss_ce = F.cross_entropy(scores, gold_targets)
        ###########################################################################
        loss = loss_ce + loss_sc / self.settings.contra_ratios
        loss.backward()
        loss = float(loss)
        ########################################################
        optimizer.step()

        return loss, loss_ce, loss_sc

    def _run_test_batch(self, batch):
        label_scores = self.model(batch.seq_lengths,
                                  batch.visual_values,
                                  batch.visual_attention_mask,
                                  batch.visual_word_starts,
                                  batch.textual_input_ids,
                                  batch.textual_attention_mask,
                                  batch.textual_word_starts,
                                  run_test=True)
        predictions = {}

        for i, size in enumerate(batch.seq_lengths):
            size = size.item()
            scores = label_scores[i, 1:size, :].cpu()
            scores = F.softmax(scores, dim=1)
            prediction = torch.argmax(scores, dim=1).float()
            predictions[batch.graph_ids[i]] = prediction

        return predictions

    def _run_train_epoch(self, data):
        self.model.train()

        total_loss = 0
        total_ce = 0
        total_sc = 0

        for i, batch in enumerate(tqdm(data)):
            batch.to(self.device)
            loss, loss_ce, loss_sc = self._run_train_batch(batch, self.optimizer)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += loss
            total_ce += loss_ce
            total_sc += loss_sc

        return total_loss, total_ce, total_sc

    def predict(self, data_path):
        print("Predicting data from", data_path)
        test_loader = self._init_test_data(data_path)
        self.model.eval()
        predictions = {}
        for batch in test_loader:
            batch.to(self.device)
            with torch.no_grad():
                pred = self._run_test_batch(batch)
                predictions.update(pred)

        return self.test_data, predictions

    def save(self, path):
        state = {"model": self.model.state_dict()}
        torch.save(state, self.settings.dir + path)

    def load(self, path):
        print("Restoring model from {}".format(path))
        state = torch.load(path)
        #####################################################################################
        resize_model_embeddings(self.model.visual_encoder, self.settings.visual_max_seq_length)
        #####################################################################################
        self.model.load_state_dict(state["model"])
        self.model = self.model.to(self.settings.device)


class ModelInteractor_ner(ModelInteractor):
    def __init__(self, settings):
        super().__init__(settings)

    def train(self):
        settings = self.settings
        print("Training is starting for {} steps using ".format(settings.step) +
              "{} with the following settings:".format(self.device))
        print()
        for key, val in settings.__dict__.items():
            print("{}: {}".format(key, val))
        print(flush=True)
        ################################################################
        train_dataloader = self._init_training_data(settings.train)
        ################################################################
        best_f1 = 0
        best_f1_epoch = 1
        early_stop = False
        ################################################################
        epoch_batch = len(train_dataloader)
        epochs = math.ceil(settings.step / epoch_batch)
        for epoch in range(1, epochs + 1):
            if not early_stop:
                print("#" * 50)
                print("Epoch:{}".format(epoch))

                total_loss, total_ce, total_sc = self._run_train_epoch(train_dataloader)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print("loss {}".format(total_loss))
                print("loss_ce:{}  loss_sc:{}".format(total_ce, total_sc))
                print('Learning_rate_textual_encoder {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                print('Learning_rate_visual_encoder {}'.format(self.optimizer.state_dict()['param_groups'][1]['lr']))
                print('Learning_rate_other {}'.format(self.optimizer.state_dict()['param_groups'][2]['lr']))
                ##################################################################
                # change learning rate
                self.scheduler.step(total_loss)
                ###########################################################
                if not settings.disable_val_eval:
                    entries, predicted, = self.predict(settings.val)
                    preds_list, out_label_list = self.align_predictions(entries, predicted)
                    f1 = f1_score(out_label_list, preds_list)
                    print("Primary Dev f1 on epoch {} is {:.2%}".format(epoch, f1))

                    improvement = f1 > best_f1
                    elapsed = epoch - best_f1_epoch

                    if not improvement:
                        print("Have not seen any improvement for {} epochs".format(elapsed))
                        print("Best f1 was {:.2%} seen at epoch #{}".format(best_f1, best_f1_epoch))
                        if elapsed == 20:
                            early_stop = True
                            print("!!!!!!!!!!!!!!!!!!!!early_stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    else:
                        best_f1 = f1
                        best_f1_epoch = epoch
                        print("Saving {} model".format(best_f1_epoch))
                        self.save("best_model.save")
                        print("Best f1 was {:.2%} seen at epoch #{}".format(best_f1, best_f1_epoch))
        self.save("last_epoch.save")

    def align_predictions(self, entries, predicted):
        batch_size = len(entries)
        label_map = dict(zip(self.settings.target_label.values(), self.settings.target_label.keys()))
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for entry in entries:
            pred = predicted[entry[0]].numpy()
            label = entry[1].numpy()
            for j in range(len(pred)):
                preds_list[entry[0] - 1].append(label_map[pred[j]])
                out_label_list[entry[0] - 1].append(label_map[label[j]])

        return preds_list, out_label_list


class ModelInteractor_pos(ModelInteractor):
    def __init__(self, settings):
        super().__init__(settings)

    def train(self):
        settings = self.settings
        print("Training is starting for {} steps using ".format(settings.step) +
              "{} with the following settings:".format(self.device))
        print()
        for key, val in settings.__dict__.items():
            print("{}: {}".format(key, val))
        print(flush=True)
        ################################################################
        train_dataloader = self._init_training_data(settings.train)
        ################################################################
        best_acc = 0
        best_acc_epoch = 1
        early_stop = False
        ################################################################
        epoch_batch = len(train_dataloader)
        epochs = math.ceil(settings.step / epoch_batch)
        for epoch in range(1, epochs + 1):
            if not early_stop:
                print("#" * 50)
                print("Epoch:{}".format(epoch))

                total_loss, total_ce, total_sc = self._run_train_epoch(train_dataloader)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print("loss {}".format(total_loss))
                print("loss_ce:{}  loss_sc:{}".format(total_ce, total_sc))
                print('Learning_rate_textual_encoder {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                print('Learning_rate_visual_encoder {}'.format(self.optimizer.state_dict()['param_groups'][1]['lr']))
                print('Learning_rate_other {}'.format(self.optimizer.state_dict()['param_groups'][2]['lr']))
                ##################################################################
                # change learning rate
                self.scheduler.step(total_loss)
                ###########################################################
                if not settings.disable_val_eval:
                    entries, predicted, = self.predict(settings.val)

                    correct = 0
                    all_item = 0
                    for entry in entries:
                        pred = predicted[entry[0]].numpy()
                        label = entry[1].numpy()
                        correct += (pred == label[:len(pred)]).sum().item()
                        all_item += len(pred)
                    acc = correct / all_item

                    print("Primary Dev acc on epoch {} is {:.2%}".format(epoch, acc))
                    improvement = acc > best_acc
                    elapsed = epoch - best_acc_epoch

                    if not improvement:
                        print("Have not seen any improvement for {} epochs".format(elapsed))
                        print("Best acc was {:.2%} seen at epoch #{}".format(best_acc, best_acc_epoch))
                        if elapsed == 20:
                            early_stop = True
                            print("!!!!!!!!!!!!!!!!!!!!early_stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    else:
                        best_acc = acc
                        best_acc_epoch = epoch
                        print("Saving {} model".format(best_acc_epoch))
                        self.save("best_model.save")
                        print("Best acc was {:.2%} seen at epoch #{}".format(best_acc, best_acc_epoch))
        self.save("last_epoch.save")
