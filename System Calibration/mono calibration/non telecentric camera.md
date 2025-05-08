# camera model
ref to MVG chapter 6
## pinhole model
* image plane/focal plane
* optical center/camera center
* principal axis/principal ray
* principal point
```
(X,Y,Z)->(fX/Z,fY/Z)
x=PX
P=[
    [f,0,0,0],
    [0,f,0,0],
    [0,0,1,0]
]
```
* P:homogenous camera projection matrix
* principal point offset
```
x=K[I|0]X_cam
K=[
    [f,0,px],
    [0,f,py],
    [0,0,1]
]
```
* world coordinate frame and camera coordinate frame
```
x=KR[I|-C~]X_world
```
* internal/intrinsic parameters, external/extrinsic parameters
```
P=K[R|t],t=-RC~
```
* focal length and principal point in terms of pixel dimensions.
```
K=[
    [f/Sx,0,px/Sx],
    [0,f/Sy,py/Sy],
    [0,0,1]
]
```
* skew parameter
* finite projective camera


# intrinsic matrix
```
[
    [fx,0,cx],
    [0,fy,cy],
    [0,0,1]
]
```

fx,fy为在像素坐标系下的等效焦距，fx=f/dx,fy=f/dy，f为物理焦距（mm），dx，dy为像素在x，y方向上的物理尺寸（mm/像素），又称作像元尺寸Sx，Sy。

cx，cy为主点坐标。