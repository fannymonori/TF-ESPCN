mkdir Training

#pascal voc dataset
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
#tar xf VOCtrainval_11-May-2012.tar -C ./Training
#rm VOCtrainval_11-May-2012.tar

#DIV2K dataset
wget data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d ./Training
rm DIV2K_train_HR.zip