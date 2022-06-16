# Testing New Models

New models can be tested on the training set-up provided in this repo. In order to add a new model to work with the training set-up:
- Confirm that the Pytorch model that you wish to use is a python class inheriting `nn.Module`, and ensure that the model of interest expects inputs of dimension:
-- `[batch_size, 1, slice_width, slice_height]` if it's a 2D model, or 
-- `[batch_size, 1, volume_z, slice_width, slice_height]` if it's a 3D model,
 _where `volume_z` represents number of stacked slices (depth) in the 3D input_.
- Create a '.json' network configuration file for the model (or copy `UNet_2D.json`)
- In that '.json' file, Assign the value of `model` as the name of the model of interest.
- Similarly specify new keys and assign them with the values of the configurable aspects of the model, that you would like to access through this configuration file.
- Assign the value of `outputdir` as the directory under which outputs of the training run should be saved.
- Assign `outputweightfilename` as the name to be given to the weight file that will be saved.
- Assign `seed` with the randomness seed that you want for the training run.
- All existing models can be seen instantiated in the 'models' section of the `task2_main.py` script.
- Add another if-block here that checks for the model name (that was specified in the model configuration file created). 
- Under this if-block instantiate the model of interest by specifying the configurations required by it. 

	_Note that the values specified in the model configuration file can be accessed here through the `network_config` dictionary variable_.

	_Also, don't forget to add `.to(device)` while instantiating the model if you wish to run it on GPU_ 
- After the above steps, the training can be run like so:
	>python3 task2_main.py --task task2.json --network UNet_2D.json

	Instead of `UNet_2D.json`, specify the model configuration file that was created for the model of interest.
	
	_If your model takes 3D input then use `task2_3D.json` instead of `task2.json`_.
