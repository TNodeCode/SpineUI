# MOT Dataset annotations

Within your MOT project there should be a directories named `data/MOT17/<train|test>` with the following structure:

```
data/MOT17
|- <train|test>
|-|- <stack_name_1>
|-|-|- gt
|-|-|-|- gt.txt
|-|-|- img
|-|-|-|- 000001.png
|-|-|-|- 000002.png
|-|-|-|- [...]
|-|-|- seqinfo.ini
|-|- <stack_name_2>
|-|-|- [...]
|-|- <stack_name_3>
|-|-|- [...]
|-|- seqmaps.txt
```

The file `seqmaps.txt` contains the names of all stacks in the following CSV format:

```csv
name
<stack_name_1>
<stack_name_2>
[...]
```

The first line is the header line and then for each stack a new line containing only the name of the stack is added. The stack names are the same as the directory names within `<train|test>`.

The file `gt.txt` contains the ground truth in the following format:

`<frame>,<object_id>,<x0>,<y0>,<width>,<height>,<confidence>,<x>,<y>,<z>`

Here is an example of how this could look like:

```
1,3,230,479,13,11,1.0,-1,-1,-1
2,3,230,478,15,17,1.0,-1,-1,-1
2,2,225,435,14,16,1.0,-1,-1,-1
```

You can find more about the MOT17 format here: https://motchallenge.net/instructions/

The file `seqinfo.ini` contains the info about the stack in the following format:

```
[Sequence]
name=<stack_name>
imDir=img
frameRate=1
seqLen=20
imWidth=512
imHeight=512
imExt=.png
seqLength=20
```