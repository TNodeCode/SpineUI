# SpineUI

## How to buid the application

```bash
# Clone the repository
$ git clone https://github.com/tnodecode/SpineUI
$ cd SpineUI

# Build the SpineUI image
$ bash scripts/docker/build.sh

# Build the mmdetection image
$ cd repositories/mmdetection && bash scripts/docker/build-light.sh

# Run the application
$ docker compose up

# Shut down the application
$ docker compose down
```