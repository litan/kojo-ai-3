# kojo-ai-3

This *extension* provides support for machine learning and AI within Kojo.

This is the third iteration of Kojo-AI. We now have a solid foundation based on [DJL](https://djl.ai/), the *Deep Java Library*. DJL gives us good access to both Pytorch and Tensorflow (and other deep learning engines where needed). It supports very usable ND Tensors, training of arbitrarily complicated models, and fast inference. So our needs should be well covered for the foreseeable future!

We will gradually have more and more documentation and learning material, but for now, to get a feel for how Kojo-AI works, take a look at the following:

The [ai_fundamentals repo](https://github.com/litan/ai_fundamentals)

Or the following examples in this repo:
* [Tensors](examples/regression/tensors.kojo)
* [Linear Regression](examples/regression/linear-regression.kojo)
* [Non Linear Regression](examples/regression/nonlinear-regression.kojo)
* Digit Classification (Mnist)
  * [Training](examples/digit-classification/mnist_train.kojo)
  * [Testing](examples/digit-classification/mnist_test.kojo)
* [Neural Style Transfer](examples/style-transfer-art/imgToimg.kojo)
* [Face Verification](examples/face-id/face-verification.kojo)

You can check out [screenshots](examples/screenshot) of the above examples in action.

### Getting Started
To start using Kojo-AI within Kojo, just [use a published release](https://github.com/litan/kojo-ai-3/releases).

---

This repository makes use of the following material:

### Neural Style Transfer
Models and images from:
* https://github.com/naoto0804/pytorch-AdaIN
* https://github.com/emla2805/arbitrary-style-transfer
* https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization

Images from:
* https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization

### Face ID
Models from:
* https://github.com/timesler/facenet-pytorch
* https://github.com/rcmalli/keras-vggface

  
