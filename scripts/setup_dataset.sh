# manually download hm3d and objectnav

VLN_Folder="/root/Projects/VLN-Game"
DATA_Folder="/root/Projects/VLN-Game/dataset/habitat"

cd $VLN_Folder
if [ -d "data" ]; then
    rm -rf data
fi
mkdir -p data
mkdir -p data/scene_datasets
ln -s $DATA_Folder/versioned_data data/versioned_data
ln -s $DATA_Folder/versioned_data/hm3d-0.1/hm3d/ data/scene_datasets/hm3d

# unzip
cd $DATA_Folder
if [ ! -d "objectnav_hm3d_v1" ]; then
    unzip objectnav_hm3d_v1.zip
fi
if [ ! -f "vlobjectnav_hm3d.zip" ]; then
    pip install gdown
    gdown 1fhXwBuGUOhF2jjW0ThtE_6rh_P3YQClj
fi
if [ ! -d "vlobjectnav_hm3d_v4" ]; then
    unzip vlobjectnav_hm3d.zip
fi

# create datasets folder
cd $VLN_Folder
mkdir -p data/datasets
ln -s $DATA_Folder/objectnav_hm3d_v1 data/datasets/objectgoal_hm3d
ln -s $DATA_Folder/vlobjectnav_hm3d_v4 data/datasets/vlobjectnav_hm3d
