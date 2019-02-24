# how to train iris detector

## Installation

### Get the code
git clone https://github.com/weiliu89/caffe.git
cd caffe
git checkout iris_ssd

### Build the code by CMake
You'd better build caffe with python2.

If you want building caffe with python3, modify CMakeLists.txt as following:
```
-set(python_version "2" CACHE STRING "Specify which Python version to use")
+set(python_version "3" CACHE STRING "Specify which Python version to use")

-  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
+  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++11")
```

**Compile**

```
$ cd $CAFFE_ROOT
$ mkdir build
$ cd build
$ cmake ..
$ make -j8; make install
```

## Dataset Preparation

Because our annotations is csv format, so we have to convert it to pascal-voc format.

### Convert CSV format annotation to pascal-voc format

#### CSV format
```
filename,left,top,right,bottom
filename1,left1,top1,right1,bottom1
filename2,left2,top2,right2,bottom2
filename3,left3,top3,right3,bottom3
...
```

#### Pascal-voc format
```
<annotation>
  <size>
    <width>300</width>
    <height>300</height>
  </size>
  <object>
    <name>face</name>
    <difficult>0</difficult>
    <bndbox>
      <xmin>100</xmin>
      <ymin>100</ymin>
      <xmax>200</xmax>
      <ymax>200</ymax>
    </bndbox>
  </object>
  <object>
    <name>face</name>
    <difficult>0</difficult>
    <bndbox>
      <xmin>0</xmin>
      <ymin>0</ymin>
      <xmax>100</xmax>
      <ymax>100</ymax>
    </bndbox>
  </object>
</annotation>
```

#### Format convertion
**Install pascal_voc_writer**
```
sudo pip3 install pascal_voc_writer
```

**csv_to_pascal_voc.py**
```
import csv
import os
import pascal_voc_writer

def csv_to_pascal_voc(csv_filename):
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        for item in reader:

            if reader.line_num == 1:
                continue
            print(item)

            # Writer(path, width, height)
            data_home = "/home/tim/datasets/iris_dataset/SingleEye_640x480_JPG/"
            abs_path =  data_home + item[0]
            writer = pascal_voc_writer.Writer(path=abs_path, width=640, height=480, depth=1, database="iris dataset")
            # ::addObject(name, xmin, ymin, xmax, ymax)
            name = "iris"
            writer.addObject(name=name, xmin=item[1], ymin=item[2], xmax=item[3], ymax=item[4])
            # ::save(path)
            pascal_voc_filename = '/home/tim/deep_learning/caffe/data/iris_dataset_devkit/single_eye_640x480/Annotations/' + item[0].split('/')[-1].split('.jpg')[0] + '.xml'
            writer.save(pascal_voc_filename)

            cmd = "cp {0} /home/tim/deep_learning/caffe/data/iris_dataset_devkit/single_eye_640x480/JPEGImages/".format(abs_path)
            os.system(cmd)

if __name__ == '__main__':
    csv_filename = 'iris.bbox.2pts.csv'
    csv_to_pascal_voc(csv_filename)
```

After running csv_to_pascal_voc.py, images will be saved to JPEGImages directory, and XML files will be saved to Annotations directory.

#### Creating trainval.txt and test.txt in ImageSets/Main directory
trainval.txt contains training images' name which without suffix ".jpg".
test.txt contains test images' name which without suffix ".jpg".

You can use the following command to generate:

```
$ cd JPEGImages
$ ls *.jpg > ../ImageSets/Main/total_image.txt
# shuffle name list 
$ cat total_image.txt | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > trainval.txt
$ cp trainval.txt test.txt
```

At last, delete last 20% of name list in trainval.txt, and delete top 80% of name list in test.txt , so that  the ratio of training samples and test samples is 8 : 2.

### Creating lmdb database
**data director tree**

```
tim@tim-server:~/deep_learning/caffe$ tree data/iris_dataset
data/iris_dataset
├── coco_voc_map.txt
├── create_data.sh
├── create_list.sh
├── labelmap_voc.prototxt
├── test_name_size.txt
├── test.txt
└── trainval.txt

tim@tim-server:~/deep_learning/caffe$ tree data/iris_dataset_devkit/ -L 2
data/iris_dataset_devkit/
├── iris_dataset
│   └── lmdb
├── single_eye_640x480
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
└── single_eye_640x480.zip
```

**modify create_list.sh**
```
root_dir=/home/tim/deep_learning/caffe/data/iris_dataset_devkit
for dataset in trainval test
do
	...
	for name in single_eye_640x480
	do
		...
	done
done
```

**modify create_data.sh**

```
root_dir="/home/tim/deep_learning/caffe"
data_root_dir="/home/tim/deep_learning/caffe/data/iris_dataset_devkit"
dataset_name="iris_dataset"
```

**modify labelmap_voc.prototxt"**

```
item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "iris"
  label: 1
  display_name: "iris"
}
```

Set "--gray = True" in caffe/scripts/create_annoset.py. Because gray scale can reduce reference time of iris detection.

**Creating lmdb database**

```
$ ./data/iris_dataset/create_list.sh
$ ./data/iris_dataset/create_data.sh
```
You will see lmdb database in /home/tim/deep_learning/caffe/data/iris_dataset_devkit/iris_dataset/.

## Model training

**Model training command: **
```
nohup ./build/tools/caffe train \
--solver="models/ResNet10/solver.prototxt" \
--gpu 0 2>&1 | tee /home/tim/deep_learning/caffe/models/ResNet10/log/ResNet10_iris_dataset_SSD_300x300.log &
```
