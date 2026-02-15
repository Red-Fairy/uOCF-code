<h2 align="center">
  <b>Unsupervised Discovery of Object-Centric Neural Fields</b>

  <b><i>TMLR 2025</i></b>


<div align="center">
    <a href="https://arxiv.org/abs/2402.07376" target="_blank">
    <img src="https://img.shields.io/badge/arXiv:2402.07376-red"></a>
    <a href="https://openreview.net/forum?id=ScEv13W2f1" target="_blank">
    <img src="https://img.shields.io/badge/TMLR:2025-orange" alt="paper"></a>
    <!-- <a href="https://red-fairy.github.io/ZeroShotDayNightDA-Webpage/supp.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Supplementary-green" alt="supp"></a> -->
    <a href="https://red-fairy.github.io/uOCF/" target="_blank">
    <img src="https://img.shields.io/badge/Project Page-blue" alt="Project Page"/></a>
</div>
</h2>

---

This is the official repository of the paper **Unsupervised Discovery of Object-Centric Neural Fields**.

**Authors:** Rundong Luo*, Hong-Xing "Koven" Yu*, Jiajun Wu (*Equal contribution)

![teaser](https://red-fairy.github.io/uOCF/assets/images/teaser.png)

For more results, please visit our [project website](https://red-fairy.github.io/uOCF/).

## Abstract
We study inferring 3D object-centric scene representations from a single image. While recent methods have shown potential in unsupervised 3D object discovery from simple synthetic images, they fail to generalize to real-world scenes with visually rich and diverse objects. This limitation stems from their object representations, which entangle objects' intrinsic attributes like shape and appearance with extrinsic, viewer-centric properties such as their 3D location. To address this bottleneck, we propose Unsupervised discovery of Object-Centric neural Fields (uOCF). uOCF focuses on learning the intrinsics of objects and models the extrinsics separately. Our approach significantly improves systematic generalization, thus enabling unsupervised learning of high-fidelity object-centric scene representations from sparse real-world images. To evaluate our approach, we collect three new datasets, including two real kitchen environments. Extensive experiments show that uOCF enables unsupervised discovery of visually rich objects from a single real image, allowing applications such as 3D object segmentation and scene manipulation. Notably, uOCF demonstrates zero-shot generalization to unseen objects from a single real image.

## Updates
- 02/15/2026: Update the dataset organization.
- 02/05/2024: All code, data, and models are released.

## Installation
Please run `conda env create -f environment.yml` to install the dependencies.

## Datasets
We gathered four datasets (Room-Texture, Room-Furniture, Kitchen-Matte, Kitchen-Shiny) for evaluating uOCF. The datasets are available at [link (Google Drive)](https://drive.google.com/drive/folders/1v_lZhiI32rvKjUDQVb5B7KHMpNLgQ2P_?usp=drive_link).

### Dataset organization

Each file begins with the prefix `{id:05d}_sc{scene_id:04d}_az{az_id:02d}`, and suffix includes `.png` for the image file, `_RT.txt` for the camera pose, and `_intrinsics.txt` for the camera intrinsics. The focal length and principal point are normalized, i.e., $f'_x = f_x / W, f'_y = f_y / H, c'_x = c_x * 2 / W - 1, c'_y = c_y * 2 / H - 1$, where $H$ and $W$ are the height and width of the image, respectively.

When running the scripts, please specify the path to the dataset by modifying the DATAROOT on the first line.

Note that we need to generate a depth image for each input image, and the MiDAS depth estimator is used in our experiments. Refer to their repo[https://github.com/isl-org/MiDaS] for more details.

## Pretrained Models
We provide the pretrained models for uOCF. The models are available at [link (Google Drive)](https://drive.google.com/file/d/167v5JYomSxgrIyFnTNP3YnCpJQjaQHAi/view?usp=sharing).

## Testing
To test the pretrained model, please run the following command:
```
bash scripts/DATASET-NAME/test-MODE.sh
```
where DATASET-NAME is one of (Room-Texture, Room-Furniture, Kitchen-Matte, Kitchen-Shiny), and MODE is one of (plane, noplane).

## Training
(Optional, we have provided pre-trained checkpoints) Run the following command to train the model from stage 1
```
bash scripts/room-texture/train-stage1-MODE.sh
```
where MODE is one of (plane, noplane).

Note that sometimes it yields undisired results (e.g., either the foreground or the background is totally black). If this happens after training for 1 epoch (~1000 iterations), you may stop the training and re-run the command.

Then run the following command to train the model from stage 2, note that Room-Furniture use the same stage 1 model as Room-Texture.
```
bash scripts/DATASET-NAME/train-stage2-MODE.sh
```
where DATASET-NAME is one of (Room-Texture, Room-Furniture, Kitchen-Matte, Kitchen-Shiny), and MODE is one of (plane, noplane).

## Citation
If you find this work useful in your research, please consider citing:
```
@article{uOCF,
  title={Unsupervised Discovery of Object-Centric Neural Fields},
  author={Luo, Rundong and Yu, Hong-Xing and Wu, Jiajun},
  journal={TMLR},
  year={2025}
}
```

## Acknowledgement
Our code framework is adapted from [uOCF](https://github.com/KovenYu/uORF). If you find any problem, please feel free to open an issue or contact the Rundong Luo at [rl897@cornell.edu](mailto:rl897@cornell.edu).



