<br />
<div align="center">
  <a href="https://github.com/skyswpark/GEOL0069_Project">
  </a>

<h3 align="center">Summary of Validation of ERA5 ROS Data with Station Rainfall Data</h3>

</div>


### Overview:

The focus of this work was to analyze rainfall data from stations across Northern Europe and Asia to compare it with interpolated rainfall data from the ERA5 dataset. The goal was to better understand the differences between observed station rainfall and ERA5 interpolations, specifically by comparing the frequency of differences, trends over time, and regional variations. This analysis involved cleaning the data, applying specific thresholds to manage data precision, and visualizing the results to identify patterns.

Initially, a rainfall threshold of 0.1mm was applied to the ERA5 data to filter out very low values. Any rainfall values below 0.1mm were considered negligible and set to zero. However, this approach was later reverted to include 0mm values, to be able to assess whether both datasets recorded zero rainfall or if there were discrepancies when one reported rainfall and the other did not.

Subsequently, rainfall thresholds were set at a range of 0 to 1mm (with a step of 0.1mm) to compare the results from different rainfall thresholds. However, the changes in the results were insignificant because the station data rainfall was recorded with no decimal points (all integers), which made it difficult to compare with the small changes in ERA5 data from using different rainfall thresholds.

The analysis focused on four key categories of comparison between the weather station rainfall data and ERA5 interpolated data:

1.	Station RA is 0 while ERA5 has rainfall: This represents cases where the weather station recorded no rainfall, but ERA5 did.
2.	ERA5 RA is 0 while Station has rainfall: This represents cases where ERA5 recorded no rainfall, but the station did.
3.	Both RA and ERA5 are 0: This captures instances where neither the station nor ERA5 reported rainfall.
4.	Both RA and ERA5 are not 0: This indicates agreement between the two datasets, with both reporting non-zero rainfall.

The focus of this analysis was to visualize and quantify how often these discrepancies occur, across different years and stations (regions). 
<br />

### Results:

Several figures were produced to assess how discrepancies evolved over time and across regions. Initially, time series plots were created to illustrate the percentage of discrepancies for each region over several years. These plots were helpful in identifying trends but had limitations, particularly when lines overlapped, making it difficult to distinguish between different categories.

Alternative visualizations, such as bar charts and stacked area plots, were considered to better highlight the relative contributions of different discrepancy categories. Additionally, histograms were used to explore the frequency distribution of rainfall values, ensuring that data cleaning and thresholding were properly applied. The ERA5 rainfall was rounded as the station data rainfall did not have any decimal points.

ERA5 rainfall seems to underestimate rainfall compared to station data rainfall.

![image](https://github.com/user-attachments/assets/ecabee31-ab40-41a3-91f3-001b8838b8a4)

![image](https://github.com/user-attachments/assets/5d18fb06-f031-4f0e-b6b2-9db0782bf62e)
