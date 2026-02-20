# era5-hourly-to-daily

A repository storing our code to go from era5 hourly data, taken with era5cli, to daily data that is usable by ESMValTool.

### Important notes

- This code is run from a linux environment, so the `path.home()` variable might differ from yours.
- You do need correct data from a leap and non-leap year
- This code works, but it is slow and there are improvements to be made for sure.


## How to Use

1. use the [era5cli](https://era5cli.readthedocs.io/en/stable/) bash script in `era5-hourly-to-daily/data_to_be_cmorized/cmor.sh` and get the correct years and variables
2. make sure you have data that is already correct, 2 datasets per variable: 1 non-leap year and 1 leap year.
3. go into: `py_cmor.py` then from the top line to the bottom:
   - make sure the `rename_dict` is correct and has the variables you want, in 'era5': 'cmip variable name'
   - check `home_dir`
   - check the `conversion_dict`, if you are not sure what it is for your variable, you can always set it to 1 and then change accordingly when testing your results
   - In this section: `# Build time bounds` we set the time bounds for 'tas' differently this is because of [this reason](https://docs.esmvaltool.org/en/latest/recipes/recipe_hydrology.html#wflow-sbm-and-wflow-topoflex)
   - In the final 'if-block' of the function you need to check if you want the file name to be of 'Eday' or 'day' or maybe even 'CFDay'
   - then in the final block `if __name__ == "__main__":` check your home folder again
4. from the command line in `era5-hourly-to-daily/` run `py_cmor.py` with: `python py_cmor.sh &` to make sure it becomes a background task.
5. The data should then appear in the `era5-hourly-to-daily/cmorized_output` folder
6. You can check the `testing_values.ipynb` notebook to visually check your data, it also has some other built in checks
   - here you need to check the home folders again
   - and the `correct_data_dict`
   - you can change the `test_variables` beware that the plot that is generated is hardcoded to be 2x2, so 4 variables.
