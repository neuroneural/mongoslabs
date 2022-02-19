# mongoslabs
Dataloader that serves MRI images from a mongodb.

The main idea is to keep MRI images and corresponding training labels
for segmentation tasks in a `mongo` database. However, each <img src="https://render.githubusercontent.com/render/math?math=256^3"> 3D MRI
tensor even in 8 bit precision is 16Mb. `mongo`'s records cannot be
larger than this limit and we need to also store the labels of the
same dimensions. `mongoslabs` fetches and aggregates each <img src="https://render.githubusercontent.com/render/math?math=256^3">
tensor stored across 8 records, together with corresponding labels
either for gray and white matter, 104 regions atlas, or a 50 region
atlas. (The scripts to populate such collection are upcoming.)

<img src="./mongoslabs_load.gif" width="50%"/>
An example of maintaining a high utilization on 4 GPUs

# installation

Eventually, the package will be placed on `pypi`, but for now, first
clone the repo:
```
git clone git@github.com:neuroneural/mongoslabs.git
```
Then change directory to the newly cloned repository:
```
cd mongoslabs
```
And install locally by
```
pip intall -e .
```
# usage
A detailed example of how to create a dataloader using provided
dataset class and the corresponding tools is in
`scripts/usage_example.py`

Do not forget to move the batches to the GPU once obtained.
