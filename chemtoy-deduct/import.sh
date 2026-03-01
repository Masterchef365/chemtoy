mkdir kinetics
cp input.py kinetics
cp export.py kinetics
sudo docker run --rm -v ./kinetics:/mnt -it reactionmechanismgenerator/rmg:3.3.0
#-c 'conda activate rmg_env; python3 rmg.py'

