# Satellite Sim

Tool that demonstrates how satellites communicate

## TODO
* Get real antenna RTL SDR setup integrated into site
* The website will have two main screens, one is Simulation, the other is Real Life
  * Add the ability to toggle between the two screens
* Simulation:
  * The current OFDM sim page looks pretty good currently
* Real Life:
  * The Real Life will show an image of the raspberry pi antenna setup
  * It will show the live spectrum frequency plot, start by targetting 137Mhz
  * Show the location of real Meteor M3/M4 Russian Satellites
  * When one comes into range try to actually decode the images they are sending
  * Log all of the image recording events, satellite name, timestamp, image received, SNR
  * Let users view the previous recording events 
  * Add a indicator if a Satellite is actively decoding an incoming image

## Setup
1. pip install rtlsdr numpy websockets
2.  **Set udev Rules (Crucial!):** The Python `rtlsdr` library needs permission to access the USB device. You'll need to create a `udev` rule.
* Create a file: `sudo nano /etc/udev/rules.d/20.rtlsdr.rules`
* Paste the following content into the file:
            SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="adm", MODE="0666"
        * Save the file (Ctrl+O, Enter, Ctrl+X).
* Reload the rules: `sudo udevadm control --reload-rules`
* **Unplug and replug your RTL-SDR dongle** for the new rule to take effect.
python rtl_sdr_server.py