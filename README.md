# Anomaly Detection in energy consumptions in buildings

**Data Analysis** 
(all the files mentioned can be found in Outputs/DataAnalysis)

Preprocessing
	1. Since there are a considerable number of missing values in the data (see the gaps in the line in raw_data_before_imputation.png), we must perform some form of imputation. 
	2. It is important to note that the gaps are quite large, not isolated points. This will effect the imputation technique used.
	3. Simple imputation methods such as filling with average of values in the neighbourhood of the missing point do not perform well when there are larger gaps (since to fill larger gaps we would have to approximate values using values that have themselves been approximated, leading to increasing errors).
	4. So we use Dynamic Time Warping Based Imputation (paper can be found in - papers/eDTWBI and papers/DTWBI) 
	5. The code for eDTWBI can be found in in DTWBI/etwbi.py
	6. Then we will apply a number of transforms on the data - 
		1. Standardization 
		2. Removal of yearly seasonality using STL decomposition (Seasonal-Trend decomposition using LOESS) 
		3. Clustering in various ways as described in the paper and comparing the performance of models fitted on particular clusters
			1. Seperating seasons (done already for daily data)
			2. Seperating weekends and weekdays
			3. Seperating each day into different periods of activity (using similar type of clustering as in seasons, except it is applied to each day instead of months)

Points of Interest 
	1. From raw_data.png we can clearly see the yearly seasonality in the data. However we cannot predict the yearly seasonality with just one year of data, so we will have to ignore this component for now. We could try a variation of each model in which the yearly seasonality is removed via STL Decomposition and compare the results with and without removal
	2. one_week.png clearly shows the daily seasonality in the data
	3. There is a clear drop in the average usage for two consecutive days after every five days (we can deduce that these are weekends)
	4. From five_weeks.png it can be noted that there is no weekly seasonality in the data
	5. Statistics for the data are present in Outputs/DataAnalysis
