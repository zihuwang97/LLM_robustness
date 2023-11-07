import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        # basic parameters
        self.parser.add_argument(
            "--output_dir", type=str, default="./checkpoint/models", help="models are saved here"
        )
        self.parser.add_argument(
            "--pretrained_checkpoint", type=str, default=None, help="models are saved here"
        )
        self.parser.add_argument(
            "--task_name",
            default="nli-snli",
            help="Data used for training",
        )
        self.parser.add_argument(
            "--data_augmentation",
            nargs="+",
            default=None,
            help="Data augmentation methods for adversarial training",
        )
        self.parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
        self.parser.add_argument("--num_labels", type=int, default=3)

        self.parser.add_argument("--max_length", type=int, default=512)
        self.parser.add_argument("--num_workers", type=int, default=5)

        self.parser.add_argument("--lower_case", action="store_true", help="perform evaluation after lowercasing")
        
        self.parser.add_argument("--pooling", type=str, default="average")

        # training parameters
        self.parser.add_argument("--per_gpu_batch_size", default=512, type=int, help="Batch size per GPU for training.")
        self.parser.add_argument("--max_epochs", type=int, default=5)
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
        self.parser.add_argument("--momentum", type=float, default=0.9)


    def print_options(self, opt):
        message = ""
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        print(message, flush=True)
        model_dir = os.path.join(opt.output_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(opt.output_dir, "models"))
        file_name = os.path.join(opt.output_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt
        