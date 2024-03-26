# Models

SpineUI allows you to connect various object detection models via a REST API interface. In this article you will learn how you can add existing endpoints to SpineUI.

## How it works

Model interfaces are managed via the `models.yaml` configuration file in the `config` directory. The content looks like this:

```yaml
models:
  - name: Yolo Docker
    url: http://yolo:80
  - name: Yolo Local
    url: http://localhost:8000
```

You can add new models by adding a new object to the `models` key containing a unique `name` and the `url` of the REST API. If the REST API is running outside of a docker container you usually use the `localhost` domain, which refers to the network interface of your local machine. However if the REST API and the user interface both run in docker images and are connected via a docker network they can use the docker domain name system. The containers usually can be reached by their names, so if a docker container's name is yolo it can be reached via `http://yolo`.
