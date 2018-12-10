========================================================
Real-world Affective Faces (RAF) Database
========================================================


For more information about the dataset, visit the project website:

  http://whdeng.cn/RAF/model1.html

If you use the dataset in a publication, please cite the paper below:

@inproceedings{li2017reliable,
  title={Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild},
  author={Li, Shan and Deng, Weihong and Du, JunPing},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  pages={2584--2593},
  year={2017},
  organization={IEEE}
}

Please note that we do not own the copyrights to these images. Their use is RESTRICTED to non-commercial research and educational purposes.


========================
Change Log
========================

Version 1.0, released on 30/03/2017

Version 1.1, released on 12/06/2018: 
We supplement the bounding box of the target face in each image. And some mistakenly aligned images have been corrected.

========================
File Information
========================

- Images (Image)
    - Original Images (Image/original.zip)
       15339 original facial images with basic expression. See IMAGE section below for more information.
    - Aligned Images (Image/aligned.zip)
       15339 aligned facial images with basic expression. See IMAGE section below for more information.

- Annotations (Annotation)
    - Manual annotations (Annotation/manual.zip)
        5 landmark location labels and other attribute labels. See ANNOTATION section below for more information.
    - Automatic annotations (Annotation/auto.zip)
        37 landmark location labels. See ANNOTATION section below for more information.
    - Bounding box (Annotation/boundingbox.zip)
        Bounding box of the target face in each image.

- Emotion Labels (EmoLabel/list_partition_label.txt)
    Expression label of each image for training and testing set respectively. See EMOTION_LABEL section below for more information.

- Features (Feature)
    - HOG feature (Feature/HOG.mat)
        HOG feature of each image for training and testing set respectively. See FEATURE section below for more information.
    - Gabor feature (Feature/Gabor.mat)
        Gabor feature of each image for training and testing set respectively. See FEATURE section below for more information.
    - baseDCNN feature (Feature/baseDCNN.mat)
        baseDCNN feature of each image for training and testing set respectively. See FEATURE section below for more information.
    - DLP-CNN feature (Feature/DLP-CNN.mat)
        DLP-CNN feature of each image for training and testing set respectively. See FEATURE section below for more information.


=========================
IMAGE
=========================

-----------------------------------------Subfile 1: original.zip-----------------------------------------

containing 12271 training samples and 3068 testing samples.

Notes:
1. Images are named in the format of "train_XXXXX.jpg" / "test_XXXX.jpg".

-----------------------------------------Subfile 2: aligned.zip-----------------------------------------

containing 12271 training samples and 3068 testing samples after aligned. 

Notes:
1. Images are named in the format of "train_XXXXX_aligned.jpg" / "test_XXXX_aligned.jpg";
2. Images are first roughly aligned using similarity transformation according to the two eye locations and the center of mouth;
3. Images are then resized to 100*100.


=========================
ANNOTATION
=========================

-----------------------------------------Subfile 1: auto.zip-----------------------------------------

containing 37 landmark location labels detected by Face++ API for each samples.

Notes:
1. In "train_XXXXX_auto_attri.txt" / "test_XXXX_auto_attri.txt", each row presents location info (x, y) of one landmark.

-----------------------------------------Subfile 2: manual.zip-----------------------------------------

containing 5 landmark location labels (the central of two eyes, the tips of the nose and two corners of the mouth) and other attribute labels (gender, race and age) manually annotated by our experimenters.

Notes:
1. In "train_XXXXX_manu_attri.txt" / "test_XXXX_manu_attri.txt", the first five lines contain location info (x, y) of 5 landmarks.
The next three lines presents information of gender, race and age attributes respectively:
	Gender
	0: male		1: female	2: unsure
   ----------------------------------------------------------------------
	Race	
0: Caucasian		1: African-American		2: Asian
----------------------------------------------------------------------
	Age (5 ranges)	
0: 0-3	 1: 4-19		2: 20-39		3: 40-69		4: 70+
----------------------------------------------------------------------


=========================
EMOTION_LABEL
=========================
------------------------- "EmoLabel/list_patition_label.txt"-------------------------

each row: <image_name> <emotion label>

Notes:
1: Surprise
2: Fear
3: Disgust
4: Happiness
5: Sadness
6: Anger
7: Neutral


=========================
FEATURE
=========================

-----------------------------------------Subfile 1: HOG.mat-----------------------------------------

containing 4000-dimensional HOG feature of each image for training and testing set respectively.

-----------------------------------------Subfile 2: Gabor.mat-----------------------------------------

containing 4000-dimensional Gabor feature of each image for training and testing set respectively.

-----------------------------------------Subfile 3: baseDCNN.mat-----------------------------------------

containing 2000-dimensional baseDCNN feature of each image for training and testing set respectively.

-----------------------------------------Subfile 4: DLP-CNN.mat-----------------------------------------

containing 2000-dimensional DLP-CNN feature of each image for training and testing set respectively.

--------------------------------------------------------------------------------------------------------------------------------------------

Notes:

1. All features are extracted on the aligned images provided in "Image/aligned.zip"
2: Please refer to the paper "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild" for the specific extraction method of each feature.


=========================
Contact
=========================

Please contact Shan Li (queenie3@live.com) for questions about the dataset.
