# Semi-Supervised Raw-to-Raw Mapping
[Mahmoud Afifi](https://sites.google.com/view/mafifi) and [Abdullah Abuolaim](https://sites.google.com/view/abdullah-abuolaim/)

York University  


Project page of the paper [Semi-Supervised Raw-to-Raw Mapping.](https://arxiv.org/abs/2106.13883) Mahmoud Afifi and Abdullah Abuolaim. arXiv preprint arXiv:2106.13883, 2021. If you use this code, please cite our paper:
```
@article{afifi2021raw2raw,
  title={Semi-Supervised Raw-to-Raw Mapping},
  author={Afifi, Mahmoud and Abuolaim, Abdullah},
  journal={arXiv preprint arXiv:2106.13883},
  year={2021}
}
```

![teaser](https://user-images.githubusercontent.com/37669469/123860756-53886680-d8f4-11eb-95a2-f324221b26a5.jpg)

### Abstract
The raw-RGB colors of a camera sensor vary due to the spectral sensitivity differences across different sensor makes and models. This paper focuses on the task of mapping between different sensor raw-RGB color spaces. Prior work addressed this problem using a pairwise calibration to achieve accurate color mapping. Although being accurate, this approach is less practical as it requires: (1) capturing pair of images by both camera devices with a color calibration object placed in each new scene; (2) accurate image alignment or manual annotation of the color calibration object. This paper aims to tackle color mapping in the raw space through a more practical setup. Specifically, we present a semi-supervised raw-to-raw mapping method trained on a small set of paired images alongside an unpaired set of images captured by each camera device. Through extensive experiments, we show that our method achieves better results compared to other domain adaptation alternatives in addition to the single-calibration solution. We have generated a new dataset of raw images from two different smartphone cameras as part of this effort. Our dataset includes unpaired and paired sets for our semi-supervised training and evaluation. 

![main](https://user-images.githubusercontent.com/37669469/123867143-fb556280-d8fb-11eb-85b5-ba67a5863435.jpg)




### Dataset
Our dataset consists of an unpaired and paired set of images captured by two different smartphone cameras: Samsung Galaxy S9 and iPhone X. The unpaired set includes 196 images captured by each smartphone camera (total of 392). The paired set includes 115 pair of images used for testing. In addition to this paired set, we have another small set of 22 anchor paired images. See our [paper](https://arxiv.org/abs/2106.13883) for more details. 

![dataset_examples](https://user-images.githubusercontent.com/37669469/123861174-dc9f9d80-d8f4-11eb-96dd-b8ffe134f8aa.jpg)

To download the dataset, please first download the DNG files, associated metadata, and pre-computed mapping from the following [link](https://ln4.sync.com/dl/f511d78f0/ji4tb5zg-ea6jm9am-ikp8dnvt-bu75h737).

Then, run `raw_extraction.py` code. Note that the code need sthe following libs to be installed: scipy, cv2, rawpy, and numpy. Make sure that the dataset is located in the root in `dataset` directory as follows:
```
- root/
      dataset/
             paired/
             unpaired/
```

The code will generate `raw-rggb` and `vis` directory (inside each subdirectory for each camera) that include `RGGB` and `RGB` images, respectively. Note that the `vis` directory (that includes `RGB` images) is for visualization as these images are gamma compressed and saved in JPG format. For the `paired` set, the code will generate `raw-rggb` and `anchor-raw-rggb` for testing and anchor sets, respectively. 


### MIT License
