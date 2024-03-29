import os
import shutil
import sys
from argparse import ArgumentParser

import torch
from seqeval.metrics import f1_score

from model_interactor import ModelInteractor_ner

target_label = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    'B-MISC': 0,
    'I-MISC': 0
}

target_label_switch = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
}



class Logger(object):
    def __init__(self, filename="../logs/log.txt", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def update(self, filename):
        self.log.close()
        self.log = open(filename, 'a+')

    def close(self):
        self.log.close()

    def flush(self):
        pass


def get_args(forced_args=None):
    parser = ArgumentParser()
    # 重要参数
    parser.add_argument("--task", type=str, default="ner")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--contra_ratios", type=int, default=5)
    parser.add_argument("--dim_textual_encoder", type=int, default=768)
    parser.add_argument("--dim_visual_encoder", type=int, default=768)
    parser.add_argument("--dim_cat", type=int, default=768 * 2)

    parser.add_argument("--dim_out", type=int, default=1536)
    parser.add_argument("--head_num", type=int, default=6)
    parser.add_argument("--textual_max_seq_length", type=int, default=512)
    parser.add_argument("--visual_max_seq_length", type=int, default=529)
    parser.add_argument("--freeze_layer", type=int, default=6)
    parser.add_argument("--disable_val_eval",
                        help="Disables evaluation on validation data",
                        type=bool,
                        default=False
                        )
    optimizer = parser.add_argument_group("Optimizer", "Set the AdamW optimizer hyperparameters")
    optimizer.add_argument("--lr_textual_encoder", type=float, default=2e-5)
    optimizer.add_argument("--lr_visual_encoder", type=float, default=2e-5)
    optimizer.add_argument("--lr_other", type=float, default=2e-4)
    optimizer.add_argument("--beta1",
                           help="Tunes the running average of the gradient",
                           type=float,
                           default=0.9
                           )
    optimizer.add_argument("--beta2",
                           help="Tunes the running average of the squared gradient",
                           type=float,
                           default=0.999
                           )
    optimizer.add_argument("--l2",
                           help="Weight decay or l2 regularization",
                           type=float,
                           default=0.05
                           )
    parser.add_argument("--rendering_backend",
                        help="Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications "
                             "it is recommended to use the default 'pangocairo'.",
                        metavar="FILE",
                        type=str,
                        default="pangocairo"
                        )
    parser.add_argument("--render_rgb",
                        help="Whether or not to render images in RGB. RGB rendering can be useful when working with "
                             "emoji but it makes rendering a bit slower, so it is recommended to turn on RGB "
                             "rendering only when there is need for it. PyGame does not support fallback fonts so "
                             "this argument is ignored when using the PyGame backend.",
                        type=bool,
                        default=False
                        )
    parser.add_argument("--overwrite_cache",
                        help="Overwrite the cached training and evaluation sets",
                        type=bool,
                        default=False
                        )
    #####################################################################################
    parser.add_argument("--visual_model_name_or_path", metavar="FILE")
    parser.add_argument("--textual_model_name_or_path", metavar="FILE")
    parser.add_argument("--load", help="Load trained model", metavar="FILE")
    parser.add_argument("--dir", type=str, default="../experiments/try")
    parser.add_argument("--data_dir",
                        type=str,
                        default=None
                        )
    parser.add_argument("--train",
                        help="train file",
                        metavar="FILE",
                        default="../sentiment_graphs/try/train.conllu"
                        )
    parser.add_argument("--val",
                        help="validation file",
                        metavar="FILE"
                        )
    parser.add_argument("--predict_file",
                        help="Skips training and instead does predictions on given file.",
                        metavar="FILE"
                        )
    parser.add_argument("--fallback_fonts_dir",
                        help="Directory containing fallback font files used by the text renderer for PIXEL. "
                             "PyGame does not support fallback fonts so this argument is ignored when using the "
                             "PyGame backend.",
                        type=str,
                        default=None
                        )
    ##############################################################################
    parser.add_argument("--seed", help="Sets the random seed", type=int, default=111)
    parser.add_argument("--dropout_label",
                        help="Dropout to label output (before attention)",
                        type=float,
                        default=0.3
                        )

    args = parser.parse_args(forced_args)

    return args


def predict(model, to_predict):
    entries, predicted = model.predict(to_predict)

    preds_list, out_label_list = model.align_predictions(entries, predicted)
    f1 = f1_score(out_label_list, preds_list)

    print("TEST f1 is {:.2%}".format(f1))

    return True


def run_parser(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.target_label_switch = target_label_switch
    args.target_label = target_label
    if not args.dir.endswith("/"):
        args.dir += "/"

    model = ModelInteractor_ner(args)

    if args.load:
        model.load(args.load)

    if args.load is None:
        model.train()

    if not args.disable_val_eval:
        model = ModelInteractor_ner(args)
        model.load(args.dir + "best_model.save")
        predict(model, args.val)

    if args.predict_file is not None:
        model = ModelInteractor_ner(args)
        model.load(args.dir + "best_model.save")
        predict(model, args.predict_file)


if __name__ == "__main__":
    args = get_args()
    sys.stdout = Logger()
    for DATASET in ["conll2003"]:
        ###############################################################################
        visual_encoder = "pixel-base"
        textual_encoder = "xlm-roberta-base"
        SETUP = "{}+{}".format(visual_encoder, textual_encoder)
        ###############################################################################
        visual_model_path = "../model/{}".format(visual_encoder)
        textual_model_path = "../model/{}".format(textual_encoder)
        TRAIN = "../datasets/{}/train.txt".format(DATASET)
        DEV = "../datasets/{}/dev.txt".format(DATASET)
        TEST = "../datasets/{}/test.txt".format(DATASET)
        DATA_DIR = "../datasets/{}".format(DATASET)
        DIR = "../experiments/{}/{}".format(DATASET, SETUP)
        LOG_DIR = "../logs/{}/{}".format(DATASET, SETUP)
        # LOAD = "../experiments/{}/{}/best_model.save".format(DATASET, SETUP)
        ###############################################################################
        args.visual_model_name_or_path = visual_model_path
        args.textual_model_name_or_path = textual_model_path
        args.train = TRAIN
        args.val = DEV
        args.predict_file = TEST
        args.data_dir = DATA_DIR
        args.dir = DIR
        # args.load = LOAD
        ###############################################################
        if args.load is None:
            if os.path.exists(DIR):
                shutil.rmtree(DIR)
            os.makedirs(DIR)
            LOGFILE = "../logs/{}/{}/train_log.txt".format(DATASET, SETUP)
        else:
            LOGFILE = "../logs/{}/{}/test_log.txt".format(DATASET, SETUP)
        ###############################################################
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        if os.path.exists(LOGFILE):
            i = 0
            while os.path.exists(LOGFILE):
                LOGFILE = "../logs/{}/{}/train_log_{}.txt".format(DATASET, SETUP, i)
                i = i + 1
        sys.stdout.update(LOGFILE)
        ###############################################################################
        print("Running {} - {}".format(DATASET, SETUP))
        run_parser(args)
        print("$" * 50)

        sys.stdout.close()
