### training
### python3 train.py training_model dropout_rate*10 training_episode training_image_used
model="$1"
python3 train.py $model 2 30 3000

### calibration
### python3 calibrate.py training_model training_parameter 10*dropout_rate evaluation_time starting_image end_image
python3 calibrate.py $1 dr0p2_ep30_ev3000 2 3000 0 10000
