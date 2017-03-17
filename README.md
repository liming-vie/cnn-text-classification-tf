**[This code based on the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

Add one more full connected layer before the dropout layer, using the input as the of pooled features and user-defined feature.

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy
- Thrift
- Word segmentation service or tool (Replace the la client)

## Training

Print parameters:

```bash
./textcnn/train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

Use [train.sh](https://github.com/liming-vie/cnn-text-classification-tf/blob/master/train/train.sh) to auto process data and setup a thrift service. 

See example in [example.sh](https://github.com/liming-vie/cnn-text-classification-tf/blob/master/train/example.sh).

```bash
cd ./train
./train.sh $train_file $feature_file $output_dir $port
```

Test:

Use [predict.sh](https://github.com/liming-vie/cnn-text-classification-tf/blob/master/train/predict.sh) to use thrift service.
```bash
cd ./train
./predict.sh port_number
```

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
