# Running the code


## Introduction
There are many ways in which you might use the code here.
We will start with some examples that describe how to test manual configurations.

First take a look at [single_train.py](../single_train.py). 
More specifically, take a look at the main function.
When continuous parameter is set to False, it will simply train the network with some resource limit 
(time, epoch or iteration). After the training, it will compute evaluation score and log all the results.
Take a look at [continuous_train.py](../continuous_train.py), it runs almost in the same way, but now 
it trains in a loop, evaluating the network periodically. 


## Examples

### MNIST 
 
We would like to test some networks using MNIST dataset. 

1. Crete new .ini configuration file, or use the default one 
    [mnist.ini](../config/mnist_dataset/mnist.ini).
2. Run the code, potentially overwriting some of the parameters using command line arguments:

```bash
python single_train.py  --ini_file config/mnist_dataset/mnist_rnn.ini \  
                        --working_dir $HOME/EEG_Results/Mnist_Tests \
                        --budget 15 \
                        --rnn_dilation 2 \
                        --rnn_num_layers 5 \
                        --skip_mode concat \
                        --batch_size 60 \
                        --rnn_hidden_size 86 \
                        --data_path /home/chrabasp/data/mnist_tmp_2 \
                        --cosine_decay 0 \
                        --dropout_f 0.13 \
                        --dropout_h 0.13 \
                        --l2_decay 0.00000026 \
                        --lr 0.001 \
                        --weight_decay 0.000028

```

3. Wait until the network is trained 

```bash
18:08:40, 29135  INFO  experiment: Initialized experiment on metagpub
18:08:40, 29135  INFO  train_manager: New model created in /home/chrabasp/EEG_Results/Mnist_Tests/train_manager/2018_05_28__16_08_40_701164/model
18:08:40, 29135  INFO  train_manager: Number of parameters in the model 336528
18:08:40, 29135  INFO  train_manager: Data reader will use an offset:  0
18:08:40, 29135  INFO  mnist_data_reader: Downloading MNIST from the web ...
18:08:43, 29135  INFO  mnist_data_reader: Loading data ...
18:08:44, 29135  DEBUG mnist_data_reader: Using CV split cv_n: 3, cv_k: 2, start: 40000, end: 60000
18:08:45, 29135  DEBUG base_data_reader: Label 0: Number of recordings 3924, Cumulative Length 3076416
18:08:45, 29135  DEBUG base_data_reader: Label 1: Number of recordings 4563, Cumulative Length 3577392
18:08:45, 29135  DEBUG base_data_reader: Label 2: Number of recordings 3943, Cumulative Length 3091312
18:08:45, 29135  DEBUG base_data_reader: Label 3: Number of recordings 4081, Cumulative Length 3199504
18:08:45, 29135  DEBUG base_data_reader: Label 4: Number of recordings 3909, Cumulative Length 3064656
18:08:45, 29135  DEBUG base_data_reader: Label 5: Number of recordings 3604, Cumulative Length 2825536
18:08:45, 29135  DEBUG base_data_reader: Label 6: Number of recordings 3975, Cumulative Length 3116400
18:08:45, 29135  DEBUG base_data_reader: Label 7: Number of recordings 4125, Cumulative Length 3234000
18:08:45, 29135  DEBUG base_data_reader: Label 8: Number of recordings 3860, Cumulative Length 3026240
18:08:45, 29135  DEBUG base_data_reader: Label 9: Number of recordings 4016, Cumulative Length 3148544
18:08:45, 29135  DEBUG base_data_reader: Number of sequences in the dataset 40000
18:08:45, 29135  INFO  base_data_reader: Create reader processes.
18:08:48, 29135  INFO  model_trainer: Will use Extended Adam with weight_decay 2.8e-05 l2_decay 2.6e-07
18:08:48, 29135  DEBUG base_data_reader: Starting train readers.
18:08:48, 29175  DEBUG base_data_reader: New reader process is running ...
18:08:48, 29176  DEBUG base_data_reader: New reader process is running ...
18:08:48, 29135  DEBUG base_data_reader: Initialize new epoch (train)
18:08:48, 29177  DEBUG base_data_reader: New reader process is running ...
/home/chrabasp/Workspace/env/lib/python3.5/site-packages/torch/nn/modules/module.py:325: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greately increasing memory usage. To compact weights again call flatten_parameters().
  result = self.forward(*input, **kwargs)
18:09:35, 29135  DEBUG model_trainer: Train iterations done 100, loss 2.18008
18:10:20, 29135  DEBUG model_trainer: Train iterations done 200, loss 1.56005
18:11:05, 29135  DEBUG model_trainer: Train iterations done 300, loss 1.41096
18:11:50, 29135  DEBUG model_trainer: Train iterations done 400, loss 1.1011
18:12:36, 29135  DEBUG model_trainer: Train iterations done 500, loss 0.928971
18:13:21, 29135  DEBUG model_trainer: Train iterations done 600, loss 0.747434
18:14:06, 29135  DEBUG model_trainer: Train iterations done 700, loss 0.64583
18:14:51, 29135  DEBUG model_trainer: Train iterations done 800, loss 0.570156
18:15:37, 29135  DEBUG model_trainer: Train iterations done 900, loss 0.484195
18:16:22, 29135  DEBUG model_trainer: Train iterations done 1000, loss 0.411248
18:17:07, 29135  DEBUG model_trainer: Train iterations done 1100, loss 0.350991
18:17:52, 29135  DEBUG model_trainer: Train iterations done 1200, loss 0.284762
18:18:37, 29135  DEBUG model_trainer: Train iterations done 1300, loss 0.229399
18:19:22, 29135  DEBUG model_trainer: Train iterations done 1400, loss 0.204335
18:20:07, 29135  DEBUG model_trainer: Train iterations done 1500, loss 0.190758
18:20:52, 29135  DEBUG model_trainer: Train iterations done 1600, loss 0.148712
18:21:37, 29135  DEBUG model_trainer: Train iterations done 1700, loss 0.149862
18:22:23, 29135  DEBUG model_trainer: Train iterations done 1800, loss 0.145898
18:23:08, 29135  DEBUG model_trainer: Train iterations done 1900, loss 0.123437
18:23:48, 29135  INFO  model_trainer: Limit reached with 1989 iterations. Stop training.
18:23:48, 29135  INFO  base_data_reader: Trying to stop train readers ...
18:23:49, 29135  DEBUG base_data_reader: During cleanup, trying to get an element when queue is empty
18:23:49, 29175  DEBUG base_data_reader: Reader received None, finishing the process ...
18:23:49, 29175  DEBUG base_data_reader: Reader process finished.
18:23:49, 29177  DEBUG base_data_reader: Reader received None, finishing the process ...
18:23:49, 29177  DEBUG base_data_reader: Reader process finished.
18:23:49, 29176  DEBUG base_data_reader: Reader received None, finishing the process ...
18:23:49, 29176  DEBUG base_data_reader: Reader process finished.
18:23:49, 29135  DEBUG base_data_reader: Waiting on join for train reader 0.
18:23:49, 29135  DEBUG base_data_reader: Train reader joined.
18:23:49, 29135  DEBUG base_data_reader: Waiting on join for train reader 1.
18:23:49, 29135  DEBUG base_data_reader: Train reader joined.
18:23:49, 29135  DEBUG base_data_reader: Waiting on join for train reader 2.
18:23:49, 29135  DEBUG base_data_reader: Train reader joined.
18:23:49, 29135  INFO  utils: Time Statistics 901.233s
18:23:49, 29135  INFO  utils: 	Get Batch took 3.16151%
18:23:49, 29135  INFO  utils: 	Forward Pass took 74.6556%
18:23:49, 29135  INFO  utils: 	Process Metrics took 9.46452%
18:23:49, 29135  INFO  utils: 	Save States took 12.4677%
<configparser.ConfigParser object at 0x7fbfa81d3cc0>
18:24:09, 29135  INFO  train_manager: Model loaded from /home/chrabasp/EEG_Results/Mnist_Tests/train_manager/2018_05_28__16_08_40_701164/model
18:24:09, 29135  INFO  train_manager: Number of parameters in the model 336528
18:24:09, 29135  INFO  train_manager: Data reader will use an offset:  0
18:24:09, 29135  WARNING base_data_reader: For validation pass we disable: balanced, random_mode, continuous, forget_state
18:24:09, 29135  INFO  mnist_data_reader: Loading data ...
18:24:10, 29135  DEBUG mnist_data_reader: Using CV split cv_n: 3, cv_k: 2, start: 40000, end: 60000
18:24:10, 29135  DEBUG base_data_reader: Label 0: Number of recordings 1999, Cumulative Length 1567216
18:24:10, 29135  DEBUG base_data_reader: Label 1: Number of recordings 2179, Cumulative Length 1708336
18:24:10, 29135  DEBUG base_data_reader: Label 2: Number of recordings 2015, Cumulative Length 1579760
18:24:10, 29135  DEBUG base_data_reader: Label 3: Number of recordings 2050, Cumulative Length 1607200
18:24:10, 29135  DEBUG base_data_reader: Label 4: Number of recordings 1933, Cumulative Length 1515472
18:24:10, 29135  DEBUG base_data_reader: Label 5: Number of recordings 1817, Cumulative Length 1424528
18:24:10, 29135  DEBUG base_data_reader: Label 6: Number of recordings 1943, Cumulative Length 1523312
18:24:10, 29135  DEBUG base_data_reader: Label 7: Number of recordings 2140, Cumulative Length 1677760
18:24:10, 29135  DEBUG base_data_reader: Label 8: Number of recordings 1991, Cumulative Length 1560944
18:24:10, 29135  DEBUG base_data_reader: Label 9: Number of recordings 1933, Cumulative Length 1515472
18:24:10, 29135  DEBUG base_data_reader: Number of sequences in the dataset 20000
18:24:10, 29135  INFO  base_data_reader: Create reader processes.
18:24:10, 29135  INFO  model_trainer: Will use Extended Adam with weight_decay 2.8e-05 l2_decay 2.6e-07
18:24:10, 29135  DEBUG base_data_reader: Starting validation readers.
18:24:11, 29362  DEBUG base_data_reader: New reader process is running ...
18:24:11, 29363  DEBUG base_data_reader: New reader process is running ...
18:24:11, 29135  DEBUG base_data_reader: Initialize new epoch (validation)
18:24:11, 29364  DEBUG base_data_reader: New reader process is running ...
18:25:03, 29135  DEBUG model_trainer: Validation iterations done 100, loss 0.107566
18:25:42, 29135  DEBUG model_trainer: Validation iterations done 200, loss 0.0956384
18:25:42, 29135  INFO  model_trainer: Limit reached with 1 epochs. Stop the run.
18:25:42, 29135  INFO  base_data_reader: Trying to stop validation readers ...
18:25:42, 29135  DEBUG base_data_reader: Waiting on join for validation reader 0.
18:25:42, 29364  DEBUG base_data_reader: Reader received None, finishing the process ...
18:25:42, 29364  DEBUG base_data_reader: Reader process finished.
18:25:42, 29363  DEBUG base_data_reader: Reader received None, finishing the process ...
18:25:42, 29363  DEBUG base_data_reader: Reader process finished.
18:25:42, 29362  DEBUG base_data_reader: Reader received None, finishing the process ...
18:25:42, 29362  DEBUG base_data_reader: Reader process finished.
18:25:42, 29135  DEBUG base_data_reader: Validation reader joined.
18:25:42, 29135  DEBUG base_data_reader: Waiting on join for validation reader 1.
18:25:42, 29135  DEBUG base_data_reader: Validation reader joined.
18:25:42, 29135  DEBUG base_data_reader: Waiting on join for validation reader 2.
18:25:42, 29135  DEBUG base_data_reader: Validation reader joined.
18:25:42, 29135  INFO  utils: Time Statistics 91.7443s
18:25:42, 29135  INFO  utils: 	Get Batch took 5.08002%
18:25:42, 29135  INFO  utils: 	Forward Pass took 18.3245%
18:25:42, 29135  INFO  utils: 	Process Metrics took 15.7855%
18:25:42, 29135  INFO  utils: 	Save States took 58.1451%
18:25:53, 29135  INFO  single_train: Train Metrics:
18:25:59, 29135  INFO  single_train: {
  "cnt_all": 93562560,
  "cnt_end": 119340,
  "example_cnt": 38004,
  "log_acc_all": 0.41421955583622777,
  "log_acc_end": 0.9008788548573834,
  "log_acc_last": 0.9069834754236397,
  "loss": 0.6028624073709485,
  "loss_all": -2.7135976250333544,
  "loss_end": -0.5923113028820085,
  "loss_last": -0.2770741245060541,
  "recent_loss": 0.1343280219193548,
  "vote_acc_all": 0.28236501420903065,
  "vote_acc_end": 0.8506209872644985,
  "vote_acc_last": 0.9069834754236397
}
18:25:59, 29135  INFO  single_train: Validation Metrics:
18:26:02, 29135  INFO  single_train: {
  "cnt_all": 15680000,
  "cnt_end": 20000,
  "example_cnt": 20000,
  "log_acc_all": 0.5922,
  "log_acc_end": 0.97025,
  "log_acc_last": 0.97025,
  "loss": 0.10160197426099331,
  "loss_all": -2.4970470669074962,
  "loss_end": -0.09533936557769776,
  "loss_last": -0.09533936557769776,
  "recent_loss": 0.09563838959671557,
  "vote_acc_all": 0.4793,
  "vote_acc_end": 0.97025,
  "vote_acc_last": 0.97025
}
18:26:03, 29135  INFO  single_train: Script successfully finished...
```

4. The final validation accuracy after 15 minutes of training is 97%, 
for details on different dictionary fields see the description of different metric classes.

```bash
python single_evaluation.py --ini_file /home/chrabasp/EEG_Results/Mnist_Tests/train_manager/2018_05_28__16_08_40_701164/config.ini \
                            --validation_data_type test
```

```bash
18:44:40, 29984  INFO  experiment: Initialized experiment on metagpub
18:44:40, 29984  WARNING single_evaluation: Evaluation on test data set (Are you sure?)
18:44:40, 29984  WARNING single_evaluation: Running evaluation, but continuous is not 0 (Are you sure?)
18:44:40, 29984  WARNING single_evaluation: Running evaluation, but forget_state is set to 1 (Are you sure?)
18:44:40, 29984  WARNING single_evaluation: Running evaluation, but balanced is set to 1 (Are you sure?)
18:44:40, 29984  INFO  train_manager: Model loaded from /home/chrabasp/EEG_Results/Mnist_Tests/train_manager/2018_05_28__16_08_40_701164/model
18:44:40, 29984  INFO  train_manager: Number of parameters in the model 336528
18:44:40, 29984  INFO  train_manager: Data reader will use an offset:  0
18:44:40, 29984  WARNING base_data_reader: For test pass we disable: balanced, random_mode, continuous, forget_state
18:44:40, 29984  INFO  mnist_data_reader: Loading data ...
18:44:41, 29984  DEBUG base_data_reader: Label 0: Number of recordings 980, Cumulative Length 768320
18:44:41, 29984  DEBUG base_data_reader: Label 1: Number of recordings 1135, Cumulative Length 889840
18:44:41, 29984  DEBUG base_data_reader: Label 2: Number of recordings 1032, Cumulative Length 809088
18:44:41, 29984  DEBUG base_data_reader: Label 3: Number of recordings 1010, Cumulative Length 791840
18:44:41, 29984  DEBUG base_data_reader: Label 4: Number of recordings 982, Cumulative Length 769888
18:44:41, 29984  DEBUG base_data_reader: Label 5: Number of recordings 892, Cumulative Length 699328
18:44:41, 29984  DEBUG base_data_reader: Label 6: Number of recordings 958, Cumulative Length 751072
18:44:41, 29984  DEBUG base_data_reader: Label 7: Number of recordings 1028, Cumulative Length 805952
18:44:41, 29984  DEBUG base_data_reader: Label 8: Number of recordings 974, Cumulative Length 763616
18:44:41, 29984  DEBUG base_data_reader: Label 9: Number of recordings 1009, Cumulative Length 791056
18:44:41, 29984  DEBUG base_data_reader: Number of sequences in the dataset 10000
18:44:41, 29984  INFO  base_data_reader: Create reader processes.
18:44:45, 29984  INFO  model_trainer: Will use Extended Adam with weight_decay 2.8e-05 l2_decay 2.6e-07
18:44:45, 29984  DEBUG base_data_reader: Starting test readers.
18:44:45, 30024  DEBUG base_data_reader: New reader process is running ...
18:44:45, 30025  DEBUG base_data_reader: New reader process is running ...
18:44:45, 29984  DEBUG base_data_reader: Initialize new epoch (test)
18:44:45, 30026  DEBUG base_data_reader: New reader process is running ...
/home/chrabasp/Workspace/env/lib/python3.5/site-packages/torch/nn/modules/module.py:325: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greately increasing memory usage. To compact weights again call flatten_parameters().
  result = self.forward(*input, **kwargs)
18:45:17, 29984  DEBUG model_trainer: Validation iterations done 100, loss 0.0964245
18:45:17, 29984  INFO  model_trainer: Limit reached with 1 epochs. Stop the run.
18:45:17, 29984  INFO  base_data_reader: Trying to stop test readers ...
18:45:17, 29984  DEBUG base_data_reader: Waiting on join for test reader 0.
18:45:17, 30024  DEBUG base_data_reader: Reader received None, finishing the process ...
18:45:17, 30024  DEBUG base_data_reader: Reader process finished.
18:45:17, 30025  DEBUG base_data_reader: Reader received None, finishing the process ...
18:45:17, 30025  DEBUG base_data_reader: Reader process finished.
18:45:17, 30026  DEBUG base_data_reader: Reader received None, finishing the process ...
18:45:17, 30026  DEBUG base_data_reader: Reader process finished.
18:45:17, 29984  DEBUG base_data_reader: Test reader joined.
18:45:17, 29984  DEBUG base_data_reader: Waiting on join for test reader 1.
18:45:17, 29984  DEBUG base_data_reader: Test reader joined.
18:45:17, 29984  DEBUG base_data_reader: Waiting on join for test reader 2.
18:45:17, 29984  DEBUG base_data_reader: Test reader joined.
18:45:17, 29984  INFO  utils: Time Statistics 32.1129s
18:45:17, 29984  INFO  utils: 	Get Batch took 7.2014%
18:45:17, 29984  INFO  utils: 	Forward Pass took 30.8571%
18:45:17, 29984  INFO  utils: 	Process Metrics took 21.8498%
18:45:17, 29984  INFO  utils: 	Save States took 37.009%
18:45:22, 29984  INFO  single_evaluation: test Metrics:
18:45:24, 29984  INFO  single_evaluation: {
  "cnt_all": 7840000,
  "cnt_end": 10000,
  "example_cnt": 10000,
  "log_acc_all": 0.5799,
  "log_acc_end": 0.9697,
  "log_acc_last": 0.9697,
  "loss": 0.0964245051331818,
  "loss_all": -2.490646244006254,
  "loss_end": -0.09190329068899154,
  "loss_last": -0.09190329068899154,
  "recent_loss": 0.0964245051331818,
  "vote_acc_all": 0.4782,
  "vote_acc_end": 0.9697,
  "vote_acc_last": 0.9697
}
```


### MNIST Architecture Search


```bash
chrabasp@metasbat1:~/Workspace/NeuralArchitectureSearch$ cd submit/bayesian_optimization_mnist/
chrabasp@metasbat1:~/Workspace/NeuralArchitectureSearch/submit/bayesian_optimization_mnist$ ./submit_grid.sh
```

```bash
Create log file directory...
Log file directory created... 

Submitting master job

Number of CPUs per task changed to 2                        

Submitted batch job 32754
Master job submitted 

Submitting worker job

Number of CPUs per task changed to 2                        

Submitted batch job 32755
Worker job submitted

Submitting worker job

Number of CPUs per task changed to 2                        

Submitted batch job 32756
Worker job submitted

Submitting worker job

Number of CPUs per task changed to 2                        

Submitted batch job 32757
Worker job submitted

Submitting worker job

Number of CPUs per task changed to 2                        

Submitted batch job 32758
Worker job submitted

Submitting worker job

Number of CPUs per task changed to 2                        

Submitted batch job 32759
Worker job submitted

Submitting worker job

Number of CPUs per task changed to 2                        

Submitted batch job 32760
Worker job submitted

Submitting worker job

Number of CPUs per task changed to 2                        

Submitted batch job 32761
Worker job submitted
```

### Pathology detection



