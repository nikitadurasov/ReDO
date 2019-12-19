#!/bin/bash
mkdir -p ./data/flowers
mkdir -p ./results/flowers
cd ./data/flowers || return
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

tar zxvf 102flowers.tgz
tar zxvf 102segmentations.tgz

rm 102flowers.tgz 102segmentations.tgz