### LCQ

## Normalized
### Before: 7 - Day and After: 7 - Day

ca_std: 2.298
ca_mean: 2.190
ca_median: 2.120

a_std: 2.676

## Raw
### Before: 7 - Day and After: 7 - Day

ca_std: 2.299
ca_mean: 2.418
ca_mean*ca_mean: 2.241
ca_median: 2.576
ca_median*ca_median: 2.219

a_std: 2.676
a_std*a_std: 2.524

### Before: 14 - Day and After: 14 - Day

ca_std: 2.431
ca_mean: 2.662
ca_median: 2.852
ca_median*ca_median: 2.622

a_std: 2.491
a_std*a_std: 2.542

### Before: 30 - Day and After: 30 - Day

ca_std: 2.635
ca_mean: 2.778
ca_median: 2.853
ca_median*ca_median: 3.054

a_std: 2.260
a_std*a_std: 2.606

## LCQ Models

#### Model 1:

ca_std + ca_median

Explained Variance Score:
[0.05708515 0.363661   0.29242826 0.36134196]
Mean Squared Error:
[0.77492708 1.09090827 1.20729547 6.52350421]
Mean Absolute Error:
[0.69699183 0.8265559  0.85913639 1.85980203]


## KBILD

#### Model 1:

ca_mean

Explained Variance Score:
[-0.15575763  0.21239284 -0.01289115  0.12644126]
Mean Squared Error:
[611.94221932 312.05458895 353.39784047 306.2360855 ]
Mean Absolute Error:
[21.11311311 13.87532441 15.20077507 15.22694926]

#### Model 2:

ca_std

Explained Variance Score:
[-0.0587338   0.22603815 -0.10543434  0.14119176]
Mean Squared Error:
[560.56918371 306.75533794 385.94057627 301.14694609]
Mean Absolute Error:
[19.74321045 14.5046003  16.30417421 15.41084936]