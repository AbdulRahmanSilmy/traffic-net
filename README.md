# Traffic Net

This project aims to detect anomaly traffic patterns across the William R. Bennett Bridge. Images are sourced from the [DriveBC website](https://images.drivebc.ca/bchighwaycam/pub/html/www/147.html), where images are uploaded at 2-minute intervals. The goal is to use object detection models to automate the traffic count within an image. This produces time series features that can be leveraged by a time series model for further analysis of traffic data.

<figure>
<figure>
    <figure style="text-align: center;">
        <img src="images/aerial_view.jpg" alt="Traffic Estimator" width="500">
        <figcaption>Arial view of the William R. Bennett Bridge</figcaption>
    </figure>
</figure>
</figure>


## Running the pipeline

To run the entire pipeline use the following command:
```sh
docker-compose -f docker-compose.yml up --build
```

## Pipeline structure 

The pipeline has three main components:

1. [Data ingestion](/data_ingestion)
2. [Object detection](/object_detection) 
3. [Anomaly detection](/anomaly_detection/) 

Each component is run seperately using docker containers. 

## [Data ingestion](/data_ingestion/)

The data ingestion component performs the following functions:

- Fetches images to be downloaded from the [DriveBC website](https://images.drivebc.ca/bchighwaycam/pub/html/www/147.html)
- Performs image preprocessing to prepare it for object detection in the next component. 

## [Object detection](/object_detection/) 

The object detection generates raw time series data relating to traffic flow. It does so by performing the following steps:

- Use object detection to get the bounding boxes of cars on traffic images
- Save bounding box information such as incoming and outgoing traffic data to a csv

<figure style="display: flex; justify-content: space-around;">
    <div style="text-align: center;">
        <img src="images/traffic_202411151838.jpg" alt="Traffic during the day" width="400">
        <figcaption>Sample image</figcaption>
    </div>
    <div style="text-align: center;">
        <img src="images/traffic_202411151838_detected.jpg" alt="Traffic during the night" width="400">
        <figcaption>Sample image with bounding boxes</figcaption>
    </div>
</figure>

## [Anomaly detection](/anomaly_detection/) (In progress)

This is the last component of the pipeline. Here anomaly events in traffic are flagged by comparing forecasted traffic with the actual traffic data. 
The following steps are carried out in this component:
- Temporally aggregate time series data to reveal trends
- Forecast future traffic trends using a rolling average model 
- Compare the real and forecasted trends to flag anomalies (coming soon!)
- Display live trends (coming soon!)
<figure>
<figure>
    <figure style="text-align: center;">
        <img src="images/time_series_plot.png" alt="Traffic Estimator" width="1300">
        <figcaption>Time series plot of traffic data</figcaption>
    </figure>
</figure>
</figure>