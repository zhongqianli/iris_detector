@[toc]

## 安装Caffe-SSD

### 获取代码
git clone https://github.com/weiliu89/caffe.git
cd caffe
git checkout ssd

### 用CMake编译
最好用python2编译pycaffe

如果要用python3，则需要修改CMakeLists.txt:
```
-set(python_version "2" CACHE STRING "Specify which Python version to use")
+set(python_version "3" CACHE STRING "Specify which Python version to use")

-  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
+  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++11")
```

**编译**

```
$ cd $CAFFE_ROOT
$ mkdir build
$ cd build
$ cmake ..
$ make -j8; make install
```

## 数据准备
因为我们的标注文件是csv格式，需要转换为pascal-voc格式

### 转换CSV格式的标注文件为pascal-voc格式

#### CSV格式
```
filename,left,top,right,bottom
filename1,left1,top1,right1,bottom1
filename2,left2,top2,right2,bottom2
filename3,left3,top3,right3,bottom3
...
```

#### Pascal-voc格式
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

#### 格式转换
**安装pascal_voc_writer**
```
sudo pip install pascal_voc_writer
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

允许csv_to_pascal_voc.py脚本后，图像将保持到JPEGImages目录， XML文件将保存到Annotations目录。

#### 在ImageSets/Main目录下创建trainval.txt和test.txt
trainval.txt包含训练样本的名字列表，名字后面没有后缀“.jpg”。
test.txt包含测试样本的名字列表，名字后面没有后缀“.jpg”。

可以用下面的命令生成:
```
$ cd JPEGImages
$ ls *.jpg > ../ImageSets/Main/total_image.txt
# shuffle name list 
$ cat total_image.txt | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > trainval.txt
$ cp trainval.txt test.txt
```

删除trainval.txt名字列表的后20%，删除test.txt的前80%，使得训练样本与测试样本的比例为8 : 2

### 创建lmdb数据库
**data目录树**

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

**修改 create_list.sh**
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

**修改 create_data.sh**

```
root_dir="/home/tim/deep_learning/caffe"
data_root_dir="/home/tim/deep_learning/caffe/data/iris_dataset_devkit"
dataset_name="iris_dataset"
```

**修改 labelmap_voc.prototxt"**

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

**创建lmdb数据库**

```
$ ./data/iris_dataset/create_list.sh
$ ./data/iris_dataset/create_data.sh
```
可以看到lmdb数据库位于 /home/tim/deep_learning/caffe/data/iris_dataset_devkit/iris_dataset/.

## 模型训练

**模型训练命令: **
```
nohup ./build/tools/caffe train \
--solver="models/ResNet10/solver.prototxt" \
--gpu 0 2>&1 | tee /home/tim/deep_learning/caffe/models/ResNet10/log/ResNet10_iris_dataset_SSD_300x300.log &
```
