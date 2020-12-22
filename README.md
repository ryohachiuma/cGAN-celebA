# cGAN-celebA

Requirements:
pytorch==1.6.0
addict
yaml
tensorboard


Create symbolic link to the celebA dataset.
```
ln -s PATH dataset
```

To train the model, please run the following command
```
python attgan.py --config ./config/cgan.yml
```

Change config parameters to test attributes....
