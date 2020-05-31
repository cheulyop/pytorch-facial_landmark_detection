
# pytorch-facial_landmark_detection

PyTorch implementation of facial landmark detection for mobile devices.

So far, in this repo are:
* [MobileNetV1]([https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861))
* [Wing Loss]([https://arxiv.org/abs/1711.06753](https://arxiv.org/abs/1711.06753))

### Prerequisites
* pytorch 1.5.0
* python >= 3.6
* 300W dataset, preferably cropped version ([link]([https://ibug.doc.ic.ac.uk/resources/300-W_IMAVIS/]))

### Example usage
First, clone the repo, or download files to a directory,
```sh
$ git clone https://github.com/cheulyop/pytorch-facial_landmark_detection.git
```
then go to the directory and run playground.py with arguments.
```sh
$ cd <path_to_repo>
$ python ./codes/playground.py --batchsize 10 --epochs 10 --gpus 1
```
Description of playground.py
```sh
$ python ./codes/playground.py -h
usage: playground.py [-h] [--batchsize BATCHSIZE] [--epochs EPOCHS]
                     [--gpus GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
  --epochs EPOCHS
  --gpus GPUS
```

### Todo
- [x] [MobileNetV1]([https://github.com/cheulyop/pytorch-facial_landmark_detection/blob/master/codes/mobilenetv1.py](https://github.com/cheulyop/pytorch-facial_landmark_detection/blob/master/codes/mobilenetv1.py))
- [x] [Wing Loss]([https://github.com/cheulyop/pytorch-facial_landmark_detection/blob/master/codes/wingloss.py](https://github.com/cheulyop/pytorch-facial_landmark_detection/blob/master/codes/wingloss.py))
- [ ] MobileNetV2
- [ ] MobileNetV3
- [ ] Adaptive Wing Loss