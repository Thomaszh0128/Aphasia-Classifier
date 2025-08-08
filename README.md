# Aphasia-Classifier
I am a high school student from Hsinchu, Taiwan.
I used Bert model(microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) and multi-head attention model to extract semantic and POS(part of speech) and gramatic and time-related information from CHA files.
Our model can be used to determine which type of aphasia one might have with descriptions of that aphasia type and probability distribution.
Our corpus comes from Aphasia bank(NIH-NIDCD R01-DC008524).
-1-The way we used to solve the problem of the ueven distribution of corpuses between different aphasia types are described as below:
1.Adaptive learning rate : 
(1) I set a patience for learning rate, which means when the change of learning rate remain smaller than some number, then the learning rate will be higher.
(2) I created a oscillation for the learning rate using Sin function.
(3) Warmup and decay of learning rate are included
2. I mixed up corpus to overcome the challenge that some type of corpus is inadequate
-2- I also used multi-head attention model to extract important imformation and add positional-embedding to them in order to make our model has more to learn on limited corpus.
