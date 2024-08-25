#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) with wav2vec2.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder /path/to/IEMOCAP_full_release

For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf

Authors
 * Yingzhi WANG 2021
"""

import os
import sys

sys.path.append('../../../speechbrain/')

import json
import pandas as pd
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb

from core import parse_arguments


class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.wav2vec2(wavs, lens)

        # last dim will be used for AdaptiveAVG pool
        outputs = self.hparams.avg_pool(outputs, lens)

        outputs = outputs.view(outputs.shape[0], -1)

        outputs = self.modules.output_mlp(outputs)
        
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using affect as label."""
        emoid, _ = batch.emo_encoded

        """to meet the input form of nll loss"""
        emoid = emoid.squeeze(1)
        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "wav2vec2_optimizer": self.wav2vec2_optimizer,
        }

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":
    # Reading command line arguments.
    data_folder, hparams_file, audio_path, run_opts, overrides = parse_arguments(sys.argv[1:])

    # Run 5 fold cross validation
   
    directory = Path(data_folder).glob('*.csv')

    iemocap_list = []
    for file in directory:
        iemocap_list.append(pd.read_csv(file))

    sb.utils.distributed.ddp_init_group(run_opts)

    seed = 1993

    for index, session in enumerate(iemocap_list):
        #override the ouput folder to allow for kfold
        output_folder = f"results_{index}/train_with_wav2vec2/{seed}"
        overrides = {'data_folder': audio_path, 'output_folder' : output_folder}

        # Load hyperparameters file with command-line overrides.
        with open(hparams_file) as fin:
          hparams = load_hyperpyyaml(fin, overrides)

        list_duplicate = iemocap_list[:]
        if index != (len(iemocap_list)-1):
            validate = iemocap_list[index+1]
            del list_duplicate[index+1]
            del list_duplicate[index]
        else:
            validate = iemocap_list[0]
            del list_duplicate[index]
            del list_duplicate[0]
        test = session
        train = pd.concat(list_duplicate, ignore_index = True)

        validate_json_dict = {}
        test_json_dict = {}
        train_json_dict = {}
        for row_index,row in validate.iterrows():
          validate_json_dict[f'session_validate_{row_index}'] = {
              "wav": f"/{{data_root}}/{row['wav']}",
              "length": row['length'],
              "emo": row['affect'],
          }
        validate_json_file = hparams["valid_annotation"]
        # Writing the dictionary to the json file
        with open(validate_json_file, mode="w") as json_f:
            json.dump(validate_json_dict, json_f, indent=2)

        for row_index,row in test.iterrows():
          test_json_dict[f'session_test_{row_index}'] = {
              "wav": f"/{{data_root}}/{row['wav']}",
              "length": row['length'],
              "emo": row['affect'],
          }
        test_json_file = hparams["test_annotation"]
        # Writing the dictionary to the json file
        with open(test_json_file, mode="w") as json_f:
            json.dump(test_json_dict, json_f, indent=2)

        for row_index,row in train.iterrows():
          train_json_dict[f'session_train_{row_index}'] = {
              "wav": f"/{{data_root}}/{row['wav']}",
              "length": row['length'],
              "emo": row['affect'],
          }
        train_json_file = hparams["train_annotation"]
        # Writing the dictionary to the json file
        with open(train_json_file, mode="w") as json_f:
            json.dump(train_json_dict, json_f, indent=2)

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Create dataset objects "train", "valid", and "test".
        datasets = dataio_prep(hparams)

        hparams["wav2vec2"] = hparams["wav2vec2"].to(device=run_opts["device"])
        # freeze the feature extractor part when unfreezing
        if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
            hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

        # Initialize the Brain object to prepare for mask training.
        emo_id_brain = EmoIdBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
    
        # The `fit()` method iterates the training loop, calling the methods
        # necessary to update the parameters of the model. Since all objects
        # with changing state are managed by the Checkpointer, training can be
        # stopped at any point, and will be resumed on next call.
        emo_id_brain.fit(
            epoch_counter=emo_id_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

        # Load the best checkpoint for evaluation
        test_stats = emo_id_brain.evaluate(
            test_set=datasets["test"],
            min_key="error_rate",
            test_loader_kwargs=hparams["dataloader_options"],
        )