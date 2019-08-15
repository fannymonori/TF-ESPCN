# TF-ESPCN

Tensorflow implementation of ESPCN algorithm described in [1].
This project was done during the Google Summer of Code 2019 program with OpenCV [2].

To run the training:
1. Download training dataset (DIV2K [3])\
`bash download_trainds.sh`
2. Run the training for 3X scaling factor\
`python main.py --train --scale 3` \
or\
Set training images directory\
`python main.py --train --scale 3 --traindir /path/to/dir`

To run the test:\
`python3 main.py --test --scale 3`\
`python3 main.py --test --scale 3 --testimg /path/to/image`

To export file to .pb format:
1. Run the export script\
`python3 main.py --export --scale 3`

There are trained .pb files in the export folder, for 2x, 3x and 4x scaling factors.

Example:\
(1) Original picture\
(2) Bicubic scaled (3x) image\
(3) ESPCN scaled (3x) image\
![Alt text](Test/t2.png?raw=true "Original picture")
![Alt text](Out/t2_bicubic_3x.png?raw=true "Bicubic picture")
![Alt text](Out/t2_ESPCN_3x.png?raw=true "ESPCN picture")

\
References

[1] Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A., Bishop, R., Rueckert, D. and Wang, Z. 
(2019). Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
 Neural Network. Available at: https://arxiv.org/abs/1609.05158 \
[2] https://summerofcode.withgoogle.com/projects/#4689224954019840 \
[3] Agustsson, E., Timofte, R. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.
Available at: http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf \
https://data.vision.ee.ethz.ch/cvl/DIV2K/
