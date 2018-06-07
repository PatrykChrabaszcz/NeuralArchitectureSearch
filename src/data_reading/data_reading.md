## TO DO: Fix this documentation



# Data Reading module. Thoughts

* To implement a new dataset simply use BaseDataReader class.
* For a new dataset you need to implement ExampleInfo class.
    * It is used to instantiate objects that represent sequences from the dataset. Examples:
        * MNIST, single digit(sequence of pixels with 784 values).  
        * TUH dataset (preprocessed), EEG recording for a specific subject 
            (sequence of 84 000 values for all 21 input channels).
    * It provides a method to read a sub-sequence from the main sequence 
        (starting from a specified time point)
    * It stores the hidden size from the last sub-sequence which will be accessed when processing
        the next sub-sequence.
    * It is important for each sequence (represented by an object of this class) to have an unique id. 
* BaseDataReader stores all example_info objects and manages from which of them reading should be done 
* BaseDataReader starts reading sub-processes. Those readers work in the following way:
    1. See which examples are 
     
* An example_info should be created for each sequence in the dataset
* There is no need to preload the data to the memory, example_info objects
    can read the data online from the hard drive.
* For small datasets it might still be better to preload them into the memory
* DataReader runs sub-processes     
    


## Motivation





With our design we want to::

1. Work with big datasets without pre-loading them into the memory.
2. Minimize the time spent on data loading in the main thread.
3. Work with truncated BPTT if desired.
4. Correctly implement truncated BPTT for RNNs with convolutional parts.
5. Work with cross validation splits.
6. Make data reader parametrized and potentially a subject for automatic optimization.


## Approach

We define a base class for all data readers: BaseDataReader.
In this class, to represent the data, we define BaseExampleInfo class.


BaseExampleInfo class:  
* Represents a specific, single sequence from the dataset, e.g:   
    
* Each object of this class has an unique id value.
* It is used to sequentially read consecutive sub-sequences from the main sequence.
* It manages proper overlapping of sub-sequences for convolutional models (see Sub-sequence Offset section)
* Can be in one of the three states:
    * Normal - It is possible to read next sub-sequence from this sequence.
    * Done - It is not possible to read next sub-subsequence because we are already at the end of this sequence or potential next sub-sequence would have smaller than required length.
    * Blocked - It is not possible to read next sub-subsequence because previous sub-sequence was read recently and new hidden state was not registered yet.
* It stores hidden state value for truncated BPTT algorithm (see section Truncated BPTT)
* It can work in 3 different modes:
    * Random mode - Reads random sub-sequence from the main sequence. In this mode hidden state should not be stored between sub-sequences.
    * Sequential mode with random offset - Reads consecutive sub-sequences (with an appropriate offset for convolutional models).
    In each epoch starts from a different initial offset. Thus, this mode can be used for training.
    * Sequential mode without offset - Reads consecutive sub-sequences but always starts from the first time-point. Thus, can be used
    during the test time.

Sub-sequence Offset:  
We assume that models that use convolutions do not use padding. 
Padding with 0's does not make sense, we can always extract real 
signal values for sub-sequences in the middle of a bigger sequence.
Because of that input and output dimensions from convolutional part
of the model do not match and we need to introduce an offset to account 
for that. Otherwise, hidden state will not be propagated correctly. 
This is best explained with an example, see Figure 1.

![OffsetExample](/images/Offset_Example.png)
*Figure 1. Different offset values. For pure RNN models, offset should be set to 0. For models that use convolutional parts,
offset value should be set to offset = cnn_output_size - cnn_input_size*


Truncated BPTT:

Dataset class creates an example_info object for each sequence. 
Then at the start of each epoch (if epoch based training is used) each of 
those examples is initialized with an initial recurrent hidden state. Then 



## Discussion






