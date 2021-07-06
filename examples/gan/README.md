# Generative Adversarial Networks

Ported from [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

## Requirements

Install [TorchVision](https://github.com/ankane/torchvision)

```sh
gem install torchvision
```

## Implementations

- [GAN](#gan)
- [Deep Convolutional GAN](#deep-convolutional-gan)
- [Wasserstein GAN](#wasserstein-gan)

### GAN

```sh
ruby gan.rb
```

[Code](gan.rb), [Original Code](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py), [Paper](https://arxiv.org/abs/1406.2661)

### Deep Convolutional GAN

```sh
ruby dcgan.rb
```

[Code](dcgan.rb), [Original Code](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py), [Paper](https://arxiv.org/abs/1511.06434)

### Wasserstein GAN

```sh
ruby wgan.rb
```

[Code](wgan.rb), [Original Code](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py), [Paper](https://arxiv.org/abs/1701.07875)
