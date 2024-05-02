# Datasets

Managing datasets with SpineUI is straight forward. The configuration for your datasets is stored in `config/datasets.yml`.

## How it works

First of all you need to place all your images somewhere on your local disk. It is recommended to use the `./datasets` directory in this project, but you could also choose any other directory. Next you need to adapt the configuration file. In the configuration file datasets look like this:

```yaml
datasets:
  - name: my_dataset_train
    paths:
      - ./datasets/my_dataset/train/*.png
  - name: my_dataset_test
    paths:
      - ./datasets/my_dataset/test/*.png
```

For every dataset you need to add an entry to the `datasets` list. Every dataset needs to have a unique `name` and a list of `paths` where the images of that dataset are located. You can use glob expressions for images containing wildcards like `/path/image_day01_*.png`. With that syntax image directories can be used for multiple datasets. Also multiple paths can be added to a single dataset.
