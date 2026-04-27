#! /bin/bash
# 
# Split and move images into batches of 1000 images each, for faster access

#############################################################################
# 
DATA_DIR=imgs
cd $DATA_DIR
#
#############################################################################
# Add here the total number of images in the dataset
NUM_IMGS=6800
BATCH_SIZE=1000

let "N_BATCHES = $NUM_IMGS / $BATCH_SIZE"

for i in `seq 0 $N_BATCHES`
do 
    mkdir -p "batch$i"
    find . -type f -maxdepth 1 | head -n 1000 | xargs -i mv "{}" "batch$i"
done

echo "Finished!"
