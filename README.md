# FanNet: A mesh convolution operator for learning dense maps
By Güneş Sucu, Sinan Kalkan, Yusuf Sahillioğlu. (Computers & Graphics)

In this work, we introduce a fast, simple and novel mesh convolution operator for learning dense shape correspondences. Instead of calculating weights between nodes, we explicitly aggregate node features by serializing neighboring vertices in a fan-shaped order. Thereafter, we use a fully connected layer to encode vertex features combined with the local neighborhood information. Finally, we feed the resulting features into the multi-resolution functional maps module to acquire the final maps. We demonstrate that our method works well in both supervised and unsupervised settings, and can be applied to isometric shapes with arbitrary triangulation and resolution. We evaluate the proposed method on two widely-used benchmark datasets, FAUST and SCAPE. Our results show that FanNet runs significantly faster and provides on-par or better performance than the related state-of-the-art shape correspondence methods.

<img width="785" height="282" alt="pipeline" src="https://github.com/user-attachments/assets/db5fd11b-935a-404b-aea5-79876bc21ef2" />

## Link
[Paper](https://authors.elsevier.com/a/1lg0T_2EOxRGB1)

## Citation
@article{sucu2025fannet,
  title={FanNet: A mesh convolution operator for learning dense maps},
  author={Sucu, G{\"u}ne{\c{s}} and Kalkan, Sinan and Sahillio{\u{g}}lu, Yusuf},
  journal={Computers \& Graphics},
  pages={104320},
  year={2025},
  publisher={Elsevier}
}

## Classification Task
This experiment classifies meshes from the SHREC2011 dataset in to 30 categories ('ant', 'pliers', 'laptop', etc). The dataset contains 20 meshes from each category, for a total of 600 inputs. The variants of each mesh are nonrigid deformed versions of one another, which makes intrinsic surface-based methods highly effective.

As with past work, we use this dataset to test the effectiveness of our model with very small amounts of training data, and train on just 10 inputs per class, selected via random split. FanNet gets nearly perfect accuracy, without any data augmentation when HKS (Heat Kernel Signature) features are used as input.

The original dataset contained meshes of about 10,000 vertices, with imperfect mesh quality (some degenerate faces, etc). In the MeshCNN paper, these were simplified to high-quality meshes of <1000 vertices, which have been widely used in subsequent work. FanNet is tested on both variants of the dataset, with similar results on each but a small improvement on the original high-resolution data. This repositiory has code and instructions for running on either dataset.

### Prerequisites

FanNet depends on pytorch, as well as a handful of other fairly typical numerical packages. These can usually be installed manually without much trouble, but alternately a conda environment file is also provided (see conda documentation for additional instructions). These package versions were tested with CUDA 11.7 on a Linux machine with Ubuntu 20.04.4 LTS. 

 ```sh
  conda env create --name fannet -f environment.yml
  ```
The code assumes a GPU with CUDA support. FanNet has minimal memory requirements; 8 GB GPU memory should be sufficient.

### Data

  The **original SHREC11 models** can be downloaded here: https://drive.google.com/uc?export=download&id=1O_P03aAxhjCOKQH2n71j013-EfSmEp5e. The relevant files are inside that archive, in the `SHREC11_test_database_new.zip` file, which is password protected with the password `SHREC11@NIST`. We also include the `data/original/categories.txt` file in this repositiory, giving ground truth labels.

  ```sh
  unzip SHREC2011_NonRigid.zip 
  unzip -P SHREC11@NIST NonRigid/SHREC11_test_database_new.zip -d data/original/raw
  ```

  The **simplified models** from MeshCNN can be downloaded here (link from the MeshCNN authors): https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz. Despite the filename, this really is the shapes from the SHREC 2011 dataset. Extract it to the `data/simplified/raw/` directory.

  ```sh
  tar -xf shrec_16.tar.gz -C data/simplified/raw
  ```

### Training from scratch

After getting ready the data for original and simplified meshes, please go to the folder `experiments/classification_shrec11` and use the following to activate fannet environment:

```sh
conda activate fannet
```
Now we can start the training process. In order to replicate the results residing in the last two rows of Table 6 of the manuscript named "FanNet: A Mesh Convolution Operator for Learning Dense Maps", use the python commands below for original and simplified meshes respectively. As you can see input_features type should be hks and the spoke_length parameters should be set to given values in the commands.

On each training run, we generate a random train/test split with 10 training inputs per-class.

To train the models on the **original** SHREC meshes, use

```python
python classification_shrec11.py --dataset_type=original --input_features=hks --spoke_length=0.0,0.02,0.04
```

And likewise, to train on the simplified meshes

```python
python classification_shrec11.py --dataset_type=simplified --input_features=hks --spoke_length=0.0,0.1,0.2
```

There will be variance in the final accuracy, because the networks generally predict just 0-3 test models incorrectly, and the test split is randomized. Perform multiple runs to get a good sample!

**Note:** This experiment is configured to generate a random test/train split on each run. For this reason, no evaluation mode or pretrained models are included to avoid potential mistakes of testing on a set which overlaps with the train set which was used for the model.

## Correspondence Task
Will be available soon!
