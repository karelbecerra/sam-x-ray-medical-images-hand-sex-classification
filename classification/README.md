# ai-sex-class-paper
Code for Sex Classification Paper

# Folder structure
## configs
Model architectures and hyperparameters
## data
Folder containing numpy dataset
## output
Output results
### checkpoint
- last_model.pth: last execution (can be combined with 'load' parameter and usefull if training is interrupted)
--
- config_name_yyyy_mm_dd_hh_mm.pth
-- Best model so far in the execution 
## samples
## src
Source code including dataloaders, utils and models
#
> wget 
# Training
> python train.py --config config_file_name --epochs EE --batch BB
 - config     (str): config file (sex_class_dropout_03)
 - epochs  (number): Is optional. If not set epochs is read from config file
 - batch   (number): Is optional. If not set batch is read from config file
 
 # Training multiple runs
> nohup ./multi_train.sh xray_original 1 10 > runs.out 2>&1 &

