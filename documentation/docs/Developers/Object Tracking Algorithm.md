# Object Tracking Algorithm

```python
ct = CentroidTracker(
    minAppeared=1,
    maxDisappeared=1,
    maxDiff=0.7,
    iomThresh=0.3,
)

# x1, y1, x2, y2, conf
ct.update(np.array([
    [10, 100, 30, 140, 0.77],
]))
print("\nOBJECTS #1", ct.objects)
print("APPEARED #1", ct.appeared)
print("DISAPPEARED #1", ct.disappeared)
ct.update(np.array([
    [11, 101, 31, 141, 0.79],
    [201, 301, 221, 321, 0.78],
]))
print("\nOBJECTS #2", ct.objects)
print("APPEARED #2", ct.appeared)
print("DISAPPEARED #2", ct.disappeared)
ct.update(np.array([
    [202, 302, 222, 322, 0.76],
    [400, 400, 420, 420, 0.91],
]))
print("\nOBJECTS #3", ct.objects)
print("APPEARED #3", ct.appeared)
print("DISAPPEARED #3", ct.disappeared)
ct.update(np.array([
    [203, 303, 223, 323, 0.76],
    [400, 400, 420, 420, 0.91],
]))
print("\nOBJECTS #4", ct.objects)
print("APPEARED #4", ct.appeared)
print("DISAPPEARED #4", ct.disappeared)
ct.update(np.array([
    [204, 304, 224, 324, 0.76],
]))
print("\nOBJECTS #5", ct.objects)
print("APPEARED #5", ct.appeared)
print("DISAPPEARED #5", ct.disappeared)
```
