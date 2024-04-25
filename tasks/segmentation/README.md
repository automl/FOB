# Task
This task imlements the [MIT Scene Parse](http://sceneparsing.csail.mit.edu/) challenge.

## Dataset
The underlying dataset is the [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset. We obtain the dataset from [Huggingface](https://huggingface.co/datasets/scene_parse_150).

## Model
We use the smallest (b0) version of the [SegFormer](https://arxiv.org/abs/2105.15203) model proposed in [SegFormer: Simple and Efficient Scene Parsing with Transformers](https://arxiv.org/abs/2105.15203).

## Performance
The metric used to evaluate the model is the mean [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index) (mIoU). We use the [implementation from mmsegmentaion](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/evaluation.html?highlight=iou%20metric#ioumetric). Our model achieves an mIoU of `35.63`. The search grid used to find the (currently) best hyperparameters can be found [here](../../baselines/segmentation.yaml). Since this task is very sensitive to the choice of learning rate, we might be able to improve this. 

### Performance comparison
We compare our performance against the pretrained model found [here](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512). The model achieves an mIoU of `36.12`. This can be tested with the following yaml:
```yaml
task:
  name: segmentation
  output_dir_name: segmentation_reference
  model:
    use_pretrained_model: true
engine:
  train: false
  plot: false
```
