# MX-Font (ICCV 2021)

**NOTICE: We release the unified few-shot font generation repository ([clovaai/fewshot-font-generation](https://github.com/clovaai/fewshot-font-generation)). If you are interested in using our implementation, please visit the unified repository.**

Pytorch implementation of ***Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Expert***. | [paper](https://arxiv.org/abs/2104.00887)

Song Park<sup>1</sup>, Sanghyuk Chun<sup>2, 3</sup>, Junbum Cha<sup>3</sup>, Bado Lee<sup>3</sup>, Hyunjung Shim<sup>1</sup><br>
<sup>1</sup> <sub>School of Integrated Technology, Yonsei university</sub>  
<sup>2</sup> <sub>NAVER AI Lab</sub>  
<sup>3</sup> <sub>NAVER CLOVA</sub>

A few-shot font generation (FFG) method has to satisfy two objectives: the generated images should preserve the underlying global structure of the target character and present the diverse local reference style. Existing FFG methods aim to disentangle content and style either by extracting a universal representation style or extracting multiple component-wise style representations. However, previous methods either fail to capture diverse local styles or cannot be generalized to a character with unseen components, e.g., unseen language systems. To mitigate the issues, we propose a novel FFG method, named Multiple Localized Experts Few-shot Font Generation Network (MX-Font). MX-Font extracts multiple style features not explicitly conditioned on component labels, but automatically by multiple experts to represent different local concepts, e.g., left-side sub-glyph. Owing to the multiple experts, MX-Font can capture diverse local concepts and show the generalizability to unseen languages. During training, we utilize component labels as weak supervision to guide each expert to be specialized for different local concepts. We formulate the component assign problem to each expert as the graph matching problem, and solve it by the Hungarian algorithm. We also employ the independence loss and the content-style adversarial loss to impose the content-style disentanglement. In our experiments, MX-Font outperforms previous state-of-the-art FFG methods in the Chinese generation and cross-lingual, e.g., Chinese to Korean, generation.

You can find more related projects on the few-shot font generation at the following links:

- [clovaai/dmfont](https://github.com/clovaai/dmfont) (ECCV'20) | [paper](https://arxiv.org/abs/2005.10510)
- [clovaai/lffont](https://github.com/clovaai/lffont) (AAAI'21) | [paper](https://arxiv.org/abs/2009.11042)
- [clovaai/mxfont](https://github.com/clovaai/mxfont) (ICCV'21) | [paper](https://arxiv.org/abs/2104.00887)
- [clovaai/fewshot-font-generation](https://github.com/clovaai/fewshot-font-generation) The unified few-shot font generation repository

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

Note that, we only provide the example font files; not the font files used for the training the provided weight *(generator.pth)*.
The example font files are downloaded from [here](https://www.freechinesefont.com/tag/commercial-use-ok/).

## Preparing Data
* The examples of datasets are in *(./data)*

### Font files (.ttf)
* Prepare the TrueType font files(.ttf) to use for the training and the validation.
* Put the training font files and validation font files into separate directories.

### The text files containing the available characters of .ttf files (.txt)
* If you have the available character list of a .ttf file, save its available characters list to a text file (.txt) with the same name in the same directory with the ttf file.
    * (example) **TTF file**: data/ttfs/train/MaShanZheng-Regular.ttf, **its available characters**: data/ttfs/train/MaShanZheng-Regular.txt
* You can also generate the available characters files automatically using the `get_chars_from_ttf.py`
```
# Generating the available characters file

python get_chars_from_ttf.py --root_dir path/to/ttf/dir
```
* --root_dir: The root directory to find the .ttf files. All the .ttf files under this directory and its subdirectories will be processed.

### The json files with decomposition information (.json)
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


## Training

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

This project is based on [clovaai/dmfont](https://github.com/clovaai/dmfont) and [clovaai/lffont](https://github.com/clovaai/lffont).

## How to cite

```
@inproceedings{park2021mxfont,
    title={Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts},
    author={Park, Song and Chun, Sanghyuk and Cha, Junbum and Lee, Bado and Shim, Hyunjung},
    year={2021},
    booktitle={International Conference on Computer Vision (ICCV)},
}
```
