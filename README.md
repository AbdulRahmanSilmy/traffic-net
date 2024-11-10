# Traffic Net

This project aims to estimate traffic across the William R. Bennett Bridge. Images are sourced from the [DriveBC website](https://images.drivebc.ca/bchighwaycam/pub/html/www/147.html), where images are uploaded at 2-minute intervals. The goal is to use object detection models to automate the traffic count within an image. This produces time series features that can be leveraged by a time series model for further analysis of traffic data.

## Data Ingestion

Data ingestion is carried out in a Docker container. The container uses the `data_ingestion.py` script within the `src` folder to periodically download images from the [DriveBC website](https://images.drivebc.ca/bchighwaycam/pub/html/www/147.html).

To run the data ingestion process, use the following command:

```sh
docker-compose -f docker-compose-rpi.yml up --build
```
