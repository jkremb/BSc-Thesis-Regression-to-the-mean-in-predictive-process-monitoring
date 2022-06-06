# Regression to the mean in predictive process monitoring

To recreate data:
1. Launch project in Anaconda with environment.yml
2. Get labeled logs from https://drive.google.com/file/d/154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR/view
3. Open experiments_notebook with Jupyter and run experiments.py with desired parameters
4. Optional: Use functions in prediction_plotter.py to graph the data


Abstract:
Predictive process monitoring involves methods of predicting outcomes of business processes, including but not limited to the application of machine learning to said business processes. Regression to the mean is a phenomenon in which sample points of a random variable tend to fall around an average value – the mean. Previous work on predictive process monitoring build machine learning models to predict business process outcomes, but remain uncertain as to whether they take regression to the mean into account. This investigation uses a predictive process monitoring framework previously built to compare the effectiveness of different parameters and extends said framework with a regression to the mean post-processing step on the predictions made. It evaluates the post-processing step through the comparison of various metrics such as AUC, Accuracy and F-Score to find whether there is a noticeable improvement of the predictions after applying regression to the mean. It finishes by concluding that there is no statistically significant change observed to assume that applying regression to the mean to a predictive process monitoring model improves its predictive accuracy.
