sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git htop vim python3-pip
sudo apt install -y rpicam-apps libcamera-tools
sudo apt install -y python3-picamera2 python3-flask python3-opencv
sudo apt install -y ffmpeg v4l2loopback-utils
sudo apt install imx500-all imx500-models

#Camera Config

sudo vim /boot/firmware/config.txt
dtoverlay=imx219,cam0
dtoverlay=imx500-pi5,cam1


sudo reboot
rpicam-hello --list-cameras

# python3
pip install mjpeg-server




#work on the camera service
vim dual_stream.py
# ---- INFILE COPY ----
cf file dual_stream.py
# ---- INFILE COPY END----



mkdir -p /home/netsaw/camera
mv /home/netsaw/dual_stream.py /home/netsaw/camera/
chmod +x /home/netsaw/camera/dual_stream.py
sudo vim /etc/systemd/system/dual_stream.service
# ---- INFILE COPY ----
[Unit]
Description=Dual camera MJPEG server
After=network-online.target
Wants=network-online.target

[Service]
User=netsaw
Group=netsaw
WorkingDirectory=/home/netsaw/camera
ExecStart=/usr/bin/python3 /home/netsaw/camera/dual_stream.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
# ---- INFILE COPY END----

sudo systemctl daemon-reload
sudo systemctl enable dual_stream.service
sudo systemctl start dual_stream.service
sudo systemctl status dual_stream.service


#work on the underclocking
lscpu | grep -E 'CPU max MHz|CPU min MHz'
# it should be
# lscpu | grep -E 'CPU max MHz|CPU min MHz'
# CPU max MHz:                             2400.0000
# CPU min MHz:                             1500.0000

sudo nano /boot/firmware/config.txt
# Under [all] (or at the end), add:
# ---- INFILE COPY ----
# Power saving
arm_freq=1200
arm_freq_min=600
gpu_freq=400
# ---- INFILE COPY END----

sudo reboot
lscpu | grep -E 'CPU max MHz|CPU min MHz'
CPU max MHz:                             1200.0000
CPU min MHz:                             600.0000
