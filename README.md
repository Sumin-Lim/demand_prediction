# Demand Prediction for Ride-hailing Platforms

**In progress**

We use spatio-temporal dataset to predict the number of rides to be matched. 
Currently prediction is made by daily frequency, based on the past 6-day logs.
Dataset should be in tablet-format and contain pickup-time, latitude and longitude of pickup-location.
Map grid is generated by Uber H3 library.

Our model file is dae.py.
