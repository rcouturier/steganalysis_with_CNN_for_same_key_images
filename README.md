# Steganalysis via a Convolutional Neural Network using Large Convolution Filters

This project show the code of the following our paper.
https://arxiv.org/1570341

The authors of this paper are:
Jean-François Couchot, Raphaël Couturier, Christophe Guyeux and Michel Salomon




We will put the pre-trained networks soon.
Likewise we will give an access of several set of images containing cover and stego images.

# Some examples

th train_stego.lua --cover images/cover_pgm  --stego images/stego_hugo_0.4  --start_test 1 --end_test 1000 --network models/model_hugo_04_iter41.t7 -p cuda 

th train_stego.lua --cover images/cover_pgm  --stego images/stego_wow_0.1  --start_test 1 --end_test 1000 --network models/model_wow_01_iter52.t7 -p cuda

th train_stego.lua --cover images/cover_jpg  --stego images/stego_uniward_0.1  --start_test 1 --end_test 1000 --network models/model_uniward_01_iter75.t7 -p cuda --ext .jpg

th train_stego.lua --cover images/cover_pgm  --stego images/stego_hugo_0.4  --start_train 1 --end_train 4000 --start_test 7001 --end_test 8000 -p cuda  -b 100

