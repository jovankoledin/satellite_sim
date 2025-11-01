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

