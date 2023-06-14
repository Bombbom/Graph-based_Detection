# Overview

## Install 

## Tutorial 

### Node Classification with DGL

- Load a DGL-provided dataset.

- Build a GNN model with DGL-provided neural network modules.

- Train and evaluate a GNN model for node classification on either CPU or GPU.

📌: Node Classification là gì?

- One of the most popular and widely adopted tasks on graph data is node classification, where a model needs to predict the ground truth category of each node. Before graph neural networks, many proposed methods are using either connectivity alone (such as DeepWalk or node2vec), or simple combinations of connectivity and the node’s own features. GNNs, by contrast, offers an opportunity to obtain node representations by combining the connectivity and features of a local neighborhood.

:pushpin: Task example

- This tutorial will show how to build such a GNN for semi-supervised node classification with only a small number of labels on the Cora dataset, a citation network with papers as nodes and citations as edges. 

### How Does DGL Represent A Graph?

- Construct a graph in DGL from scratch.

- Assign node and edge features to a graph.

- Query properties of a DGL graph such as node degrees and connectivity.

- Transform a DGL graph into another graph.

- Load and save DGL graphs.

### Write your own GNN module

- Understand DGL’s message passing APIs.

- Implement GraphSAGE convolution module by your own.

### 

### Training a GNN for Graph Classification

- Load a DGL-provided graph classification dataset.

- Understand what readout function does.

- Understand how to create and use a minibatch of graphs.

- Build a GNN-based graph classification model.

- Train and evaluate the model on a DGL-provided dataset.


### Make Your Own Dataset


