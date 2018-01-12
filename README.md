# Yield-Prediction-Using-Sentinel-Data

The motive here is to predict the yield of crops of a particular farm by the change in pixels of the image of farm yearly .
The satellite data used is of sentinel , sentinel is a launched by European Space Agency and the data is open source .

1) Sentinel Image Products Scraping - The script in this folder scraps a patch of land containing our farm of interst , this scripts                                             automatically scraps all the images through out the year .

2) Preprocessing and Cropping - The script in this folder will crop the farm which we need to analyze according to coordinates provided by                                 the user and also does feature engineering on the cropped images . 

3) Tensorflow Model - The script in this folder has the algorithm , it is still under testing ang the algorithm is somewhat similar to                           algorithm from this paper : https://cs.stanford.edu/~ermon/group/website/papers/jiaxuan_AAAI17.pdf.

In the above mentioned paper the data used is of NASA which is of low resolution as compared to that Sentinel .
