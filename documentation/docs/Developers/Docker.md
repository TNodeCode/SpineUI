# Docker Containers

## How to build the SpineUI Docker container

If you want to build the docker container for SpineUI you can do this by executing the following command in the project's root directory where `Dockerfile` is stored.

```bash
$ docker build -t tnodecode/spineui .
```

When the command has finished your docker image is ready to be used.

## Uploading the docker image to Docker Hub

If you want to make your docker image publically available you first need to create an account for Docker Hub (https://hub.docker.com). Next you need to connect Docker on your local machine to your Docker Hub account by running `docker login`. You will need to enter your password. After that you have to tag your Docker image with the following naming schema: `<username>/<image_name>`. So if your username is `johndoe` and you want to publish an image with the name `abcde` you would first need to build that image on your machine by running `docker build -t johndoe/abcde` and then uploading it to Docker Hub by running `docker push johndoe/abcde`. After the image is uploaded everyone can use your image by doownloading it via the command `docker pull johndoe/abcde`.

### Updating existing images

For updating existing images you need to perform the same steps as for uploading new images. Docker will find out which image layers in the Docker registry need to be updated and will only upload those layers to the Docker Hub registry.
