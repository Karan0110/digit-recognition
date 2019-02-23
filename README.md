# Notes

- The default ```network.ann``` supplied has an accuracy of about 93%
- The program can only deal with **28 x 28 images** (like in the MNIST database)
- **The image must have a _black background_ and a _white foreground_ (not the other way around)**

# Usage

Running the main program without command line arguments:
```
python3 recognize.py

Enter file path: IMAGE_OF_3
IMAGE_OF_3 - 3

```

Running the main program with command line arguments:
```
python3 recognize.py IMAGE_OF_3 IMAGE_OF_8 IMAGE_OF_1

IMAGE_OF_3 - 3
IMAGE_OF_8 - 8
IMAGE_OF_1 - 1
```

Changing training settings to get a (hopefully) better network:

- Open ```trainer.py```
- Change the key word arguments supplied in ```net.fit(X_train, y_train, minibatch_size=10, num_epochs=50000, learning_rate=.5, reporting=100)```

Training a new neural network (takes a while):
```
python3 trainer.py
```

# Dependencies

- numpy
- python-mnist (for parsing the MNIST data set)
