# MOTRv2 Annotations

## Detector Results Format

Results from a detection model are stored in a JSON file named `det_db_motrv2.json`. The file contains detections in the following schema:

```json
{
    "<filename1>": [
        "x0,y0,w,h,score",
        "x0,y0,w,h,score",
        ...
    ],
    "<filename2>": [
        "x0,y0,w,h,score",
        "x0,y0,w,h,score",
        "x0,y0,w,h,score",
        ...
    ],
    ...
}
```

An example could look like this:

```json
{
  "DanceTrack/train/dancetrack0016/img1/00001863.txt": [
    "1088.1,368.6,221.4,543.4,0.98\n",
    "826.2,361.1,217.4,481.3,0.97\n",
    "735.8,390.1,162.0,394.9,0.97\n",
    "495.5,324.0,149.8,614.2,0.97\n",
    "968.0,403.0,175.5,387.4,0.96\n",
    "398.9,357.8,141.1,484.7,0.92\n"
  ],
  "DanceTrack/train/dancetrack0016/img1/00001877.txt": [
    "484.3,349.0,189.0,573.1,0.98\n",
    "1119.8,371.6,180.9,538.3,0.97\n",
    "797.9,370.6,294.3,471.2,0.97\n",
    "978.8,407.0,147.2,377.3,0.95\n",
    "763.4,386.1,135.0,397.6,0.94\n",
    "411.4,361.1,144.4,482.6,0.91\n"
  ]
}
```