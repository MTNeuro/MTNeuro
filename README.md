# MTNeuro Benchmark Dataset
![dataset](assets/dataset.png)

## Links 
* [Website](https://mtneuro.github.io/)
* [Bossdb Page](https://bossdb.org/project/prasad2020)

## Overview 

We introduce a new dataset, annotations, and multiple downstream tasks that provide diverse ways to readout information about brain structure and architecture from the same image. Our multi-task neuroimaging benchmark (MTNeuro) is built on volumetric, micrometer-resolution X-ray microtomography imaging of a large thalamocortical section of mouse brain, encompassing multiple cortical and subcortical regions, that reveals dense reconstructions of the underlying microstructure (i.e., cell bodies, vasculature, and axons). We generated a number of different prediction challenges and evaluated several supervised and self-supervised models for brain-region prediction and pixel-level semantic segmentation of microstructures. Our experiments not only highlight the rich heterogeneity of this dataset, but also provide insights into how self-supervised approaches can be used to learn representations that capture multiple attributes of a single image and perform well on a variety of downstream tasks.

## Getting started
Installation
Requirements
Running an example, expected output
Pytorch Dataloader
Non-pytorch downloads

## Code structure
The main code for the package is found in the MTNeuro folder. 

## Training Scripts
Several existing benchmarks are provided. You can also add and benchmark your own algorithm. 

## License 
This software is available under the [MIT License](https://opensource.org/licenses/MIT) 

The X-ray Microtomography image dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). 

## Citation
If you find this project useful in your research, please cite the following papers:

* Prasad, J. A., Balwani, A. H., Johnson, E. C., Miano, J. D., Sampathkumar, V., De Andrade, V., ... & Dyer, E. L. (2020). A three-dimensional thalamocortical dataset for characterizing brain heterogeneity. Scientific Data, 7(1), 1-7.