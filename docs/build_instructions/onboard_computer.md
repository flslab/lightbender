# Setup Raspberry Pi CM5

## Install OS

## Install Offboard Controller

If cloning a private GitHub repository, you need to first authorize the RPi in the repository setting
1. Create an SSH key on the RPi:
```commandline
ssh-keygen -t ed25519 -C lbXX
```
The comment is optional and for better organization. XX is the ID of the LightBender, e.g., lb02.
Press enter for the rest of inputs.
2. In the GitHub repository, go to Settings > Deploy Keys.
3. Add deploy key and pase the content of `.ssh/id_ed25519.pub`.
```commandline
cat .ssh/id_ed25519.pub
```

Clone the repository using the SSH command, and then run the setup script.
```commandline
git clone git@github.com:flslab/fls-cf-offboard-controller.git
cd fls-cf-offboard-controller
bash setup.sh
```

This script automatically setups a Python environment, install dependencies, and configures the RPi.

Reboot after setup is completed.

## Test LED
Turns on 50 LEDs for 5 seconds.
```commandline
python led.py -t 5 -n 50
```

## Test Servo
```commandline
python servo_pwm.py -i -n 1
```
Then test different angles by inputting servo index and angle. For example 0 0 or 0 90.

## Test USB connection with CF Bolt
```commandline
python controller.py --ground-test
```