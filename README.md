# VLN-Game


Use Replica-Dataset in Habitat-Lab:

```
cd Dataset/
mkdir replica_v1
cd ../
cd Replica-Dataset/
./download.sh ../Dataset/replica_v1/
sudo apt-get update
sudo apt-get install pigz
./download.sh ../Dataset/replica_v1/
cd ..
cd Dataset/
unzip sorted_faces.zip 
cd sorted_faces/
./copy_to_folders ../replica_v1/
```
