[![DeepSource](https://deepsource.io/gh/compSPI/simSPI.svg/?label=active+issues&show_trend=true&token=9eFu6aig3-oXQIuhdDoYTEq-)](https://deepsource.io/gh/compSPI/simSPI/?ref=repository-badge)

# simSPI
Methods and tools for simulating SPI data.

## Contributing

See our [contributing](https://github.com/compspi/compspi/blob/master/docs/contributing.rst) guidelines!

## Questions for wrapper

- Write utility functions in ioSPI allowing to go from one data format to another - utility functions that convert pdb to cif to 3d or accept all 3 ?
- Ask if classy?
- Ask configuration format? 
- seperate input file for pdb,cif?
- do we care about interim files?

## Notes

**Main functions**

Parameters:

TEMSimulator(inputfile,configuration_yaml,output_mrc=None,output_config=None) -> output: numpy data

Output:

TEMSimulator(inputfile,configuration_yaml) -> output : simulation object

simulation object


key/property : type
data: np array,
configuration: object,
write_mrc: function(output_file,config = true)


**Utility functions**



- generate_simulation_file_dirs(input_file, output_mrc,output_config) //(cryoemio.simio() but accepts all formats including cif,3d maps)
- fill_parameters_dictionary(yaml_file)
- fill_grid_in_fov(sample_dimensions, optics_params,detector_params, input_file=input_file,Dmax=30, pad=5.) 
                                        

**Work flow**

configurations -> 
define-grid (write crd file) -> 
build dictionary -> 
build input file from dictionary ->
run simulator ->
get mrc 
