# Testing New Models Using Pytorch for Task 2

The Pytorch dataset can be adapted for training in existing pytorch scripts following the examples in the script folder.

Alternatively, and to compare directly against all existing baselines, new models can quickly be trained and tested using the Pytorch training set-up provided in this repo. In order to add a new Pytorch model to the training set-up:
* Confirm that the Pytorch model that you wish to use is a python class inheriting `nn.Module`, and ensure that the model of interest expects inputs of dimension:
	* `[batch_size, 1, slice_width, slice_height]` if it's a 2D model, or 
	* `[batch_size, 1, volume_z, slice_width, slice_height]` if it's a 3D model, where `volume_z` represents number of stacked slices (depth) in the 3D input.
* Create a '.json' network configuration file for the model (or copy `UNet_2D.json`)
* In that '.json' file, Assign the value of `model` as the name of the model of interest.
* Similarly specify new keys and assign them with the values of the configurable aspects of the model, that you would like to access through this configuration file.
* Assign the value of `outputdir` as the directory under which outputs of the training run should be saved.
* Assign `outputweightfilename` as the name to be given to the weight file that will be saved.
* Assign `seed` with the randomness seed that you want for the training run.
* All existing models can be seen instantiated in the 'models' section of the `task2_main.py` script.
	* Add another block for the new model name (that was specified in the model configuration file created). 
	* Under this block, instantiate the model of interest by specifying the configurations required. 
* After the above steps, the training can be run as: ```python3 task2_main.py --task task2.json --network UNet_2D.json```
* If your model takes 3D input then use `task2_3D.json` instead of `task2.json`.
