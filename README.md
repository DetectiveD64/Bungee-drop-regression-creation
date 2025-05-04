# Bungee-drop-regression-creation
Create a regression + equation for Science Olympiad Bungee drop data

very basic python code
put drop data in array format under the # Given data 
[length of cord used , mass used] for X
[length of cord extension] for Z

Uses second degree regression because it is assumed that length of cord used and mass should have a linear relationship with with both.

Make sure to get drop data on endpoints; highest height + highest mass, highest height + lowest mass, lowest mass + lowest height, etc.

Extrapolation is extremely inconsistent/inaccurate (approximately 7x worse based on testing)

Code reates regression equation and graph.
Prints:
- RMSE
- MAE
- R^2 value
Measures of error/deviation

I was able to get MAEs of <3 cm.

scores at the Science Olympiad NY states competition were below 30 for top ten. 
This means an average of 15 cm off the target on both drops, easily achievable with the regression.
Good luck to anyone

