nohup ./build/tools/caffe train \
--solver="models/MobileNet-SSD/solver_train.prototxt" \
--gpu 0 2>&1 | tee /home/tim/deep_learning/caffe/models/MobileNet-SSD/log/MobileNet_iris_dataset_SSD_300x300.log &
