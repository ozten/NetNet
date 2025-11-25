# AGENT

We want our code to be as portable as possible between Apple MPS and CUDA.

## Guidelines

* Always specify `touch.float32` and never use `touch.float64` when creating a tensor.
  * This is because MPS does not support float64.
* We do not use `torch.compile` which isn't stable on MPS.