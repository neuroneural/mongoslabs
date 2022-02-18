# mongoslabs
Dataloader that serves MRI images from a mogodb.

The main idea is to keep MRI images and corresponding training labels
for segmentation tasks in a `mongo` database. However, each <img src="https://render.githubusercontent.com/render/math?math=256^3"> 3D MRI
tensor even in 8 bit precision is 16Mb. `mongo`'s records cannot be
larger than this limit and we need to also store the labels of the
same dimensions. `mongoslabs` fetches and aggregates each <img src="https://render.githubusercontent.com/render/math?math=256^3">
tensor stored across 8 records, together with corresponding labels
either for gray and white matter, 104 regions atlas, or a 50 region
atlas. (The scripts to populate such collection are upcoming.)

# installation

# usage
