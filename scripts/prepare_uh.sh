#!/bin/bash

#======================================================
# Prepare UCF-HMDB dataset
# Note: change the paths based on your actual paths
#======================================================

set -e 

DEST_DIR=/ssd_scratch/cvit/avijit/

mkdir -p $DEST_DIR
rsync -av --progress=info2 avijit.d@ada:/share3/dataset/ucf101-hmdb51/ucf-hmdb.tar.gz $DEST_DIR

cd $DEST_DIR
 
tar -xf ucf-hmdb.tar.gz

mkdir -p $DEST_DIR/logs

cd -