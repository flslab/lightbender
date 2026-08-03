This is Version 0 of the software for airflow sensor.  The python file in this directory is a standalone software executed on a RPi.  

Hardware:  
- Manufacturer:  Sensirion AG
- Model Number:  SEK-SDP31
- Description:   Digital Pressure Sensor Evaluation Kit.
- URL:  https://www.digikey.com/en/products/detail/sensirion-ag/SEK-SDP31/16517716?gclsrc=aw.ds&gad_source=1&gad_campaignid=20243136172&gbraid=0AAAAADrbLlj0im6ZqZtGK14O_kQQsEUze&gclid=CjwKCAjwvNfSBhBiEiwAyaGMCZABxsizYITPo_NiJU_WP1Pk6Mj4sWkzWIMcSKoZKAWlVQoowsPHaxoCNSIQAvD_BwE

3D Printed Parts: 
- Mounting of the airflow sensor to the rod for conducting experiments:
  * 3D printable files are Flow_element_grip.3mf and Rod_grip.3mf
  * 3D fusion files are FlowElement_grip.f3d - unfortunately, the fusion file for the other one is lost. 
- Prior to the arrival of the sensor, we 3D printed the part in order to setup the experiments with LightBenders.  The AutoCAD Fusion version of these files is ./

Software:  Custom built python at USC.

Requirements:
1. Enable I2C on RPi by running sudo raspi-config in the terminal.  Navigate to Interface Options, select I2C, choose Yes to enable it and automatically load kernel modules, and then reboot the system.  
2. Update ../orchestrator.py with the updated logic in ./orchestrator.py. 
3. Copy the manifest file (./swarm_manifest_sample.yaml) that describes the airflow sensor node in the Orchestrator directory.  
4. Run the updated Orchestrator with the additional flag, using the following command:  Python3 orchestrator.py --illumination --sensor

Output:
1. The revised Orchestrator generates log records based on Experiments 1 to 4 in the Google Slide. 

Experiments conducted using this software in the context of four different experiments are available in the following Google Slide:  https://docs.google.com/presentation/d/1wRDwtCE3NHDeBA4fOQYP56STGkDCzMYEMwvgELqPWFg/edit?usp=sharing
