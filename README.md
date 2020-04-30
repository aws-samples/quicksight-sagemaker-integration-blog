## ML Predictions using Amazon QuickSight and Amazon SageMaker

At re:Invent 2019, Amazon QuickSight announced Amazon SageMaker integration, a new feature that allows you to integrate your own SageMaker machine learning models with QuickSight, to analyze the augmented data, and use it directly in your business intelligence dashboards. As a business analyst, with just a few clicks, you can perform machine learning inference in QuickSight, a process to make predicting on new data, using SageMaker models for different use cases such as predicting likelihood of customer churn, scoring leads to prioritize sales activity and assessing credit risk for loan applications. 

Traditionally, getting the predictions from trained models into a BI tool requires substantial heavy lifting. This requires the data engineering team to write code to ETL the data into S3, call the inference API to get the predictions, ETL the model output from S3 to a queryable source and orchestrating this process whenever new data is available, not to mention repeating workflow for every single model that you build. Setting up inference endpoints enables easier consumption of the models, but comes at a cost of potentially incurring charges when not in use. Finally, a business analyst will use 3rd party tool like Excel to perform analysis and reporting on the predictions, making it difficult to update and share the results to a large number of users. The end-to-end process to get value from the model is undifferentiated and can take days or weeks before the predictions of the models get to the business decision makers.

QuickSight’s integration with Amazon SageMaker addresses these pain points by automating the entire data movement and inference pipeline. 


To run inferencing on your dataset, you can connect to any of the QuickSight supported data sources (e.g., S3, Athena, Aurora, RDS and Amazon Redshift, as well as 3rd party application sources like Salesforce, ServiceNow, and JIRA), select the pretrained SageMaker model you want to use for prediction and QuickSight takes care of the rest. Once the data ingestion and inference job are complete, you can create visualizations and build reports based on the predictions and share it with the business stakeholders, all in an end-to-end workflow. The output of the inference is stored as a QuickSight SPICE dataset and makes it possible to share it with others. You can perform manual, one-off predictions, run predictions on a schedule or trigger predictions programmatically via the QuickSight dataset APIs when your data refreshes.

We use an example of churn where we predict likelihood of customers leaving a mobile phone operator. Imagine as the provider, if you find out which customer is leaving, you can offer timely incentives such as phone upgrade or perhaps have a new feature activated, and you might keep your customer around. Incentives are often much more cost effective than losing and reacquiring a customer. We use customer interaction and service usage data to build such model that can find those most likely to churn. We show you how to train a SageMaker model to predict customer churn in this repo. 

To build and train the model, first [create an Amazon SageMaker notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html). Next, load this repo onto the newly created notebook instance, then open the jupyter notebook contained in the quicksight-sagemaker-integration folder and follow the instructions in the notebook.

The [blog post](https://aws.amazon.com/blogs/TBA/) associated with this repo is located here.


## License

This library is licensed under the MIT-0 License. See the LICENSE file.

