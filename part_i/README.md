## Part I


#### Data Download

I download 200 tickers where 20 are supposed to be a train set. \
Some other than stock market series were taken as well.

#### Data Preprocessing

* Data Splitting
* Feature Encoding
* Adding Simple Features
    * Especially some statistics which are lost after normalization
* Normalization

#### Clustering

* KMeans
* KMeans with DTW metric

#### Feature Extraction

* FFT (it doesn't work well)
* Wavelet Transform
* Statistical Features with tsfel

#### Training

* Wrapper feature selection
* Random Forest
* CatBoost