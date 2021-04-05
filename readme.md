# Introduction

Pytorch implementation of ***Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Expert***.

* * *

## Prerequisites

* **Python > 3.6**

  Using conda is recommended: [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)
* **pytorch >= 1.5**

	To install: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
	

* sconf, numpy, scipy, scikit-image, tqdm, jsonlib, fonttools

```
conda install numpy scipy scikit-image tqdm jsonlib-python3 fonttools
```


# Usage
The examples of datasets are in *(./data)*
Note that, we only provide the example font files; not the font files used for the training the provided weight *(generator.pth)*.
The example font files are downloaded from [here](https://www.freechinesefont.com/tag/commercial-use-ok/).

## Training
### Preparing font files (.ttf)
* Prepare the TrueType font files(.ttf) to use for the training and the validation.
* Put the training font files and validation font files into separate directories.
* The files for the decomposition information are needed.
    * The files for the Chinese characters are provided. (data/chn_decomposition.json, data/primals.json)
    * If you want to train the model with a language other than Chinese, the files for the decomposition rule (see below) are also needed.
        * **Decomposition rule**
            * structure: dict *(in json format)*
            * format: {char: [list of components]}
            * example: {'㐬': ['亠', '厶', '川'], '㐭': ['亠', '囗', '口']}
        * **Primals**
            * structure: list *(in json format)*
            * format: [**All** the components in the decomposition rule file]
            * example: ['亠', '厶', '川', '囗', '口']


### Modify the configuration file (cfgs/train.yaml)

```
- use_ddp:  whether to use DataDistributedParallel, for multi-GPUs.
- port:  the port for the DataDistributedParallel training.

- work_dir:  the directory to save checkpoints, validation images, and the log.
- decomposition:  path to the "decomposition rule" file.
- primals:  path to the "primals" file.

- dset:  (leave blank)
  - train:  (leave blank)
    - data_dir : path to .ttf files for the training
  - val: (leave blank)
    - data_dir : path to .ttf files for the validation
    - source_font : path to .ttf file used as the source font during the validation

```

### Run training
```
python train.py cfgs/train.yaml
```
* **arguments**
	* path/to/config (first argument): path to configration file.
	* \-\-resume (optional) : path to checkpoint to resume.


### Test

### Preparing the reference images
* Prepare the reference images and the .ttf file to use as the source font.
* The reference images are should be placed in this format:

```
    * data_dir
    |-- font1
        |-- char1.png
        |-- char2.png
        |-- char3.png
    |-- font2
        |-- char1.png
        |-- char2.png
            .
            .
            .
```

* The names of the directory or the image files are not important, however, **the images with the same reference style are should be grouped with the same directory.**
* If you want to generate only specific characters, prepare the file containing the list of the characters to generate.
    * The example file is provided. (data/chn_gen.json)
    
### Modify the configuration file (cfgs/eval.yaml)

```
- dset:  (leave blank)
  - test:  (leave blank)
    - data_dir: path to reference images
    - source_font: path to .ttf file used as the source font during the generation
    - gen_chars_file: path to file of the characters to generate. Leave blank if you want to generate all the available characters in the source font.

```
    
### Run test
```
python eval.py \
    cfgs/eval.yaml \
    --weight generator.pth \
    --result_dir path/to/save/images
```
* **arguments**
  * path/to/config (first argument): path to configration file.
  * \-\-weight : path to saved weight to test.
  * \-\-result_dir: path to save generated images.
  
## Code license

This project is distributed under [MIT license](LICENSE), except [modules.py](models/modules/modules.py) which is adopted from https://github.com/NVlabs/FUNIT.

```
MX-Font
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Acknowledgement

This project is based on [clovaai/dmfont](https://github.com/clovaai/dmfont).
