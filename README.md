## Introduction

*This repo was cloned & edited from https://github.com/NVlabs/stylegan2.*

This project aims to generate formal passport photo from informal photos with human face. It includes 500_128_passport dataset which consists of over 500 passport photos of size 128x128 ready to train.

**Requirements**

This repo was tested on Ubuntu 18.04 and Tensorflow 1.14 with GPU support enabled.

```.bash
pip3 install -r requirements.txt
```

**Training**

Already trained model is in the results folder. If it is not present, please download from https://drive.google.com/drive/folders/1UbUAbGyZInt6Qt4PDI2s6YR8SAfexuUh

To generate your own model:

```.bash
python3 run_training.py --num-gpus=1 --data-dir=datasets --config=config-f --dataset=500_128_passport
```

**Encoding**

Put images you want to convert into raw_images/ dir and run the following command. Generated results will be in the generated_images/ directory.

```.bash
python3 encode.py
```

and then

```.bash
python3 translate.py
```
