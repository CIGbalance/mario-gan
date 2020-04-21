# mario-gan
Mario Level Generation as an optimisation problem for the GBEA benchmark

Data here is in compiled format (jar), source code is released here: https://github.com/TheHedgeify/DagstuhlGAN/tree/v1.0

```
cd /this/repository
java -Djava.awt.headless=true -jar $PWD/dist/MarioGAN.jar "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" $PWD/GAN/overworld-10-5000/netG_epoch_4999_5641.pth 10 1 0 1
```
