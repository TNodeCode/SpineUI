# SpineUI

## Initialize the SpineUI project on your system

```bash
# Clone the repository
$ git clone https://github.com/tnodecode/SpineUI
$ cd SpineUI
```

## How to buid the docker images for your application

```bash
# Build the SpineUI image
$ bash scripts/docker/build.sh

# Build the mmdetection image
$ cd repositories/mmdetection && bash scripts/docker/build-light.sh
```

## Download the datasets and model weights

There are two ZIP files: `datasets.zip` and `work_dirs.zip`. For now ask the authors of this project for these files.

The first ZIP file should be extracted to `./datasets` and the second one to `./work_dirs`.

## Run the application

When you finished with the steps above you can run the application by executing the following command from the root directory of this project (where the docker-compose.yaml file is located).

```bash
# Run the application
$ docker compose up
```

# Shut down the application

When you want to shut down the application ht CTRL+C / Command+C in your terminal and then run the following command:

```bash
# Shut down the application
$ docker compose down
```