#!/bin/bash

RealTestDirs=("../forensic-transfer/FFHQ/test ../forensic-transfer/celebA/test ../forensic-transfer/faceforensics_images/FaceForensics_source_to_target_images/test/faceforensics_real ")

# for dir in $RealTestDirs
# do
# 	echo $dir
# done

echo $1

# echo 'testing real datasets'
# for dir in $RealTestDirs
# do
# echo $dir
# if [ $dir == '../forensic-transfer/celebA/test' ]
# then
# 	python train.py --test True --trained_model $1 --folder $dir --test_type real --data_type celeba 
# else
# 	python train.py --test True --trained_model $1 --folder $dir --test_type real 
# fi
# done

echo 'testing fake datasets'
FakeTestDirs=("../forensic-transfer/pggan_fake/test ../forensic-transfer/stargan/test ../forensic-transfer/faceforensics_images/FaceForensics_source_to_target_images/test/faceforensics_fake ../forensic-transfer/FaceSwap/FaceSwap/test ../forensic-transfer/faceforensicspp/deepfake_test/cropped_squared/ ../forensic-transfer/stylegan-master/test ")
for dir in $FakeTestDirs
do
echo $dir
python train.py --test True --trained_model $1 --folder $dir --test_type fake 
done
