# TF-ESPCN

Tensorflow implementation of ESPCN algorithm described in [1].

To run the training:
1. Download training dataset\
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

\
References

[1] Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A., Bishop, R., Rueckert, D. and Wang, Z. 
(2019). Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
 Neural Network. Available at: https://arxiv.org/abs/1609.05158 \
 