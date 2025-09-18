CLIC - Common Line Implied Clustering
Donovan Webb & Yuriy Chaban

Clustering procedure based off of 3D heterogeneity of particle
projections based off of the common line theorem.

Inputs: Extracted particles.
Outputs: (Hopefully) Homogeneous clusters of particles. Can also
inspect dendrograms from each batch to see progressive grouping of
particles. If using Relion output star file can be used to perform 3D
reconstruction.

Accepted inputs are a directory of individual particles in mrc format,
a particle stack in .mrcs format or a Relion star file of extracted
particles.

Pipeline of CLIC:
- Particle input.
- Sinograms made from particles.
- Each line of each sinogram is used as independent input to dimensional
  reduction.
- Dimensionally reduced lines are scored on 'closeness'.
- Similar lines heirarchically clustered leading to particles being
  clustered.
- Dendrogram cut to produce clusters.


Running the job:

usage: main.py -i DATA_SET [-n NUM] [-b BATCH_SIZE] [-d DOWN_SCALE]
       [-c NUM_COMPS] [-m MODEL] [-l NLINES] [-g] [-k NUM_CLUSTERS] [-r SNR]

 -h, --help            show this help message and exit

 -i DATA_SET, --data_set DATA_SET
                       Dataset to be considered for clustering. Input
                       path to mrcs stack, to individual mrc
                       particles, or to particle starfile with
                       "/PATH/TO/PARTS/*.mrc" (Notes: 1. Don't forget
                       "", 2. if star file: run from relion home dir
                       3. expects *.mrc or path/to/file.mrcs or
                       path/to/file.star only.

 -n NUM, --num NUM     Number of projections to consider total. Defaults to 1000

 -b BATCH_SIZE, --batch_size BATCH_SIZE
                       Batchsize - runs overlapping batches of
                       provided size. This speeds up process and
                       requires less memory. Recommended batch size is
                       750 < b < 2000

 -d DOWN_SCALE, --down_scale DOWN_SCALE
                       Downscaling of image prior to making sinograms

 -c NUM_COMPS, --num_comps NUM_COMPS
                       Number of components of dimensional reduction
                       technique. This requires some experimentation

 -m MODEL, --model MODEL
                       Dimensional reduction technique. options are:
                       PCA, UMAP, TSNE, LLE, ISOMAP, MDS,
                       TRIMAP. Recommended: UMAP and PCA.

 -l NLINES, --nlines NLINES
                       Number of lines in one sinogram (shouldn't need
                       to change recommended=120)

 -g, --gpu             Run on gpu with CUDA

 -k NUM_CLUSTERS, --num_clusters NUM_CLUSTERS
                       Number of clusters

 -r SNR, --snr SNR
		       For testing: Signal to noise ratio to be
		       applied to projection before making sinograms


example:

$ main.py -i "/Path/To/PARTICLES/*.mrc" -n 10000 -b 1000 -m UMAP -c 10 -g -k 6

Will run CLIC on 10,000 particles in overlapping batches of 1000
particles each. Dimensional reduction performed by UMAP with 10
components and final dendrogram 'cut' to produce 6 clusters.
