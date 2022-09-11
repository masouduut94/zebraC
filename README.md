# QUIZ answers
# Before running other notebooks make sure you have these items prepared.
- directory named `videos` containing videos from `1.mp4` to `10.mp4`

## install these packages beforehand
- pip install git+https://github.com/okankop/vidaug.git
- pip install mediapy
- pip install opencv-python
- pip install torch, torchaudio, torchvision
- pip install gensim
- pip install scikit-learn
- pip install nltk
- pip install pandas
- pip install numpy

# There are two jupyter notebooks (they are both answers):


# `VideoCaption.ipynb` has used tf-idf Vectorizer for text generation.

## Q1: Explain why did you design the video dataloader in this way? 
The dataloader loads two parts. 
- Part 1: Samples N frames from each video and returns Batch\*N\*H\*W\*C of video which is used to train 3d CNNs or convLSTM (or attention-based networks) network to extract spatiotemporal features.
- Part 2: Reads the captions and after preprocessing the text, Sentences would be converted into vectors by `tf-idf Vectorizer` and then it is gonna be fed to a network with LSTM shape.


## Q2. What are the weaknesses of your video loader?

- 1: One of the bottlenecks of dataloader is using `opencv` to iterate through the video and sample the frames. this makes the process so slow.
- 2: It is a good practice to cache data in numpy format and read them using torch functions to make the process speed up.
- 3: In case we have big amount of text data, it is not good to train the Vectorizer each time we initialize the `dataloader`. It's also not good to keep all the vectors in the RAM, which could be costly for training process.



# `VideoCaption2.ipynb` has used `keras.preprocessing.text.Tokenizer` and `keras.preprocessing.sequence.pad_sequences`

## Q: Explain why did you design the video dataloader in this way? 

The dataloader loads two parts. 
- Part 1: Samples N frames from each video and returns Batch\*N\*H\*W\*C of video which is used to train 3d CNNs or convLSTM (or attention-based networks) network to extract spatiotemporal features.
- Part 2: Reads the captions and after preprocessing the text, Sentences would be converted into vectors to be fed into transformer-alike CNNs.
