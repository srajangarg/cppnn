# prereqs
sudo apt install clang-format-3.8

# compile
Run `make`
It will produce 4 executables
- test -> Runs Neural Network on MNIST Data without CUDA
- test_cuda -> Runs Neural Network on MNIST Data with CUDA
- test_conv -> Runs Image Convolution without CUDA
- test_conv_cuda -> Runs Image Convolution with CUDA

# run
Neural Network can be run by just running the executables.
It will show errors and time on stdout
`./test
./test_cuda`

Convolution can be done using following commands
`./test_conv [image-path]
./test_conv_cuda [image-path]`

This will generate images with applied effect with same image name append with the effect

# links
https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm

# workflow
```
git checkout -b new_feature
...
git push origin new_feature
(create PR by going to https://github.com/srajangarg/cppnn)
```

