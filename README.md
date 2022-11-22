# Unsupervised Image Segmentation

In this repository I re-implement from scratches the code for unsupervised image segmentation described in the paper

Wonjik Kim\*, Asako Kanezaki\*, and Masayuki Tanaka.
**Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering.** 
*IEEE Transactions on Image Processing*, accepted, 2020.
([arXiv](https://arxiv.org/abs/2007.09990))

or in the [GitHub repository](https://github.com/kanezaki/pytorch-unsupervised-segmentation). 

## Novel Contributions

Besides re-organising the entire process, I have added (and will keep adding with time) a number of features. 

- It is now possible to train on multiple images, and not just one. This might make sense if we have multiple very similar images (or for any other application you might see suited). 
- I have generalised the loss function: in addition to the continuity and similarity loss functions presented in the paper, I have added a contribution coming from very big and very small areas. This is to ensure that creation of too wide or too narrow segmentation is disfavoured. 
- Each component of the loss function is weighted by a parameter that can be set through the parser. 

## Run

To get started, place a (possibly small) colour image inside the input directory. Then, run the script with

`python main.py <FILENAME>`

## Options

There are various options available:

- `--path <path>`: pass the input directory images should be read from. Default is `Input`;
- `--n_channels <3 or 1>`: number of channels (colours) of input images;
- `--n_features <integer>`: dimension of latent space after CNN;
- `--n_classes <integer>`: number of classes (distinct regions) the image is initially segmented with;
- `--min_classes <integer>`: minimum number of classes (distinct regions) the image can be segmented into;
- `--n_convolutions <integer>`: number of convolutional steps to be applied to the image;
- `--batch_size <integer>`: batch size;
- `--learning_rate <float>`: learning rate for the SGD optimizer;
- `--momentum <float>`: momentum for the SGD optimizer;
- `--epochs <integer>`: epochs for training;
- `--mu <float>`: float for weight of similarity loss function;
- `--nu <float>`: float for weight of continuity loss function (overall);
- `--nu_x <float>`: float for weight of continuity loss function along x;
- `--nu_y <float>`: float for weight of continuity loss function along y;
- `--M <float>`: float for weight of max area loss function;
- `--m <float>`: float for weight of min area loss function;
- `--model_path <path>`: path for saved model;
- `--show`: show the results of segmentation as the training runs.
