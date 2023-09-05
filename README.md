# wandb Tutorial

wandb is a free tool for logging data from machine learning training processes. It includes features for user management, team management, and project management.

[[Zhihu Tutorial (in Chinese)](https://www.zhihu.com/column/c_1494418493903155200)] [[中文介绍](./README_zh.md)]

## 0. Environment Setup

- Ubuntu or Red Hat
- tmux
- Python3
- `pip install -r requirements.txt`

## 1. Basic Usage

For a detailed tutorial on this section, see: [wandb Usage Tutorial (Part 1): Basic Usage](https://zhuanlan.zhihu.com/p/493093033)
- Example of showing training curves: [test_curves.sh](./basic/test_curves.sh)
- Example of showing images: [test_images.sh](./basic/test_images.sh)
- Example of showing videos: [test_videos.sh](./basic/test_videos.sh)
- Example of showing matplotlib plots: [test_matplot.sh](./basic/test_matplot.sh)
- Example of showing tables: [test_tables.sh](./basic/test_tables.sh)
- Example of showing multi-process groups: [test_multi_process.sh](./basic/test_multi_process.sh)
- Example of showing HTML: [test_html.sh](./basic/test_html.sh)
- Example of PyTorch integration: [test_pytorch.sh](./basic/test_pytorch.sh)

## 2. Hyperparameter Search

For a detailed tutorial on this section, see: [wandb Usage Tutorial (Part 2): Distributed Hyperparameter Search Using Launchpad](https://zhuanlan.zhihu.com/p/496164470)

In machine learning tasks, we often encounter many hyperparameters that need tuning. wandb provides features for hyperparameter search. However, wandb primarily focuses on hyperparameter search scheduling and visualization, and does not inherently offer distributed capabilities. Thus, this section describes a way to combine [Launchpad](https://github.com/deepmind/launchpad) and wandb for parallel (or distributed) hyperparameter search.

Note: Since Launchpad doesn't offer multi-machine distributed capabilities, if you wish to perform multi-machine parallel hyperparameter searches, consider using [TLaunch](https://github.com/TARTRL/TLaunch).

- Basic example: [test_sweep.sh](./sweep/launchpad/test_sweep.sh), this example provides a minimal setup for combining wandb and Launchpad.
- Search for dropout hyperparameter in CNN classification task: [test_sweep.sh](./sweep/cnn/test_sweep.sh), this example is based on the MNIST classification task and focuses on searching for the best dropout parameter.

## 3. Data and Model Management

wandb also offers features for data and model backup management. For a detailed tutorial on this section, see: [wandb Usage Tutorial (Part 3): Data and Model Management](https://zhuanlan.zhihu.com/p/503226955)

- Basic example: [test_artifact.sh](./artifact/test_artifact.sh), this example provides a way to back up MNIST training data.

## 4. Local Deployment of wandb

wandb also offers features for local server deployment. For a detailed tutorial on this section, see: [wandb Usage Tutorial (Part 4): Local Deployment of wandb](https://zhuanlan.zhihu.com/p/521663928)

## Citing wandb_tutorial

If you use wandb_tutorial in your work, please cite us:

```bibtex
@article{huangshiyu2022wandb,
    title={wandb Tutorial},
    author={Shiyu Huang},
    year={2022},
    howpublished={\url{https://github.com/huangshiyu13/wandb_tutorial}},
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=huangshiyu13/wandb_tutorial&type=Date)](https://star-history.com/#huangshiyu13/wandb_tutorial&Date)
