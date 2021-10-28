# It includes input arguments and options used in system
from configparser import ConfigParser
import torch


class Options:
    def __init__(self, config_file_name):
        """Parse all sections of config file

        Parameters
        ----------
        config_file_name: str
            input configuration file
        """
        self.config = ConfigParser()
        self.config.read(config_file_name)

        self.experiment_name = self.config["DEFAULT"]["experiment_name"]
        device_name = self.config["DEFAULT"]["device"]
        if device_name == 'gpu':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.source_dataset_name = self.config["DEFAULT"]["source_dataset_name"]
        self.target_dataset_name = self.config["DEFAULT"]["target_dataset_name"]
        self.batch_size = int(self.config["DEFAULT"]["batch_size"])
        self.only_source = self.config["DEFAULT"].getboolean("only_source")

        self.learning_rate = float(self.config["DEFAULT"]["learning_rate"])
        self.optimizer = self.config["DEFAULT"]["optimizer"]
        self.momentum = float(self.config["DEFAULT"]["momentum"])
        self.weight_decay = float(self.config["DEFAULT"]["weight_decay"])
        self.loss = []
        for loss in (self.config["DEFAULT"]["loss"]).split('+'):
            loss_weight, loss_name = loss.split('*')
            self.loss.append({'loss_weight': float(loss_weight), 'loss_name': loss_name})

        self.epoch_num = int(self.config["DEFAULT"]["epoch_num"])
        self.log_step = float(self.config["DEFAULT"]["log_step"])


# # Test options
# if __name__ == '__main__':
#
#     params = Options('../config.ini')
#     tmp = 0