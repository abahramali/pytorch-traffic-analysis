# pytorch-traffic-analysis

This repository contains an unofficial PyTorch implementations of state-of-the-art deep nerual network based traffic analysis techniques. 

## Automated Website Fingerprinting (AWF)
Original paper ([PDF](https://arxiv.org/abs/1708.06376)): 

```
Vera Rimmer, Davy Preuveneers, Marc Juarez, Tom Van Goethem, and Wouter Joosen. Automated Website Fingerprinting through Deep Learning. In 2018 Network and Distributed System Security Symposium (NDSS). 
```

<!-- ### Implementation
-  AWF used  -->

### Dataset
The original AWF website contains 900 websites each with 2500 samples. For the implementation is this repository, we used a subset of this websites with 100 websites and 2500 traces for each of them. 
You can download the dataset [here](https://distrinet.cs.kuleuven.be/software/tor-wf-dl/files/tor_100w_2500tr.npz).


## Deep Fingerprinting (DF)
Original paper ([PDF](https://dl.acm.org/doi/abs/10.1145/3243734.3243768))

```
Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning. In 2018 ACM SIGSAC Conference on Computer and Communication Security (CCS). 
```

### Implementation
The implementation in this repository is for the closed-world scenarion without any defense mechanisms.

### Dataset
The DF dataset contains 95 websites each with 800 samples for training, 100 samples for validation, and 100 samples for test. You can download the dataset [here](https://drive.google.com/drive/folders/1kxjqlaWWJbMW56E3kbXlttqcKJFkeCy6).

## VarCNN 
Original paper ([PDF](https://arxiv.org/pdf/1802.10215.pdf))

```
Sanjit Bhat, David Lu, Albert Kwon, and Srinivas Devadas. Var-CNN: A Data-Efficient Website Fingerprinting Attack Based on Deep Learning. In 2019 Privacy Enhancing Technologies (PETs).
```

### Dataset
You can find the original dataset of the paper in their [repository](https://github.com/sanjit-bhat/Var-CNN).

## Triplet Fingerprinting
Original paper ([PDF](https://dl.acm.org/doi/abs/10.1145/3319535.3354217))

```
Payap Sirinam, Nate Mathews, Mohammad Saidur Rahman, and Matthew Wright. Triplet Fingerprinting: More Practical and Portable Website Fingerprinting with N-shot Learning. In 2019 ACM SIGSAC Conference on Computer and Communication Security (CCS).
```

### Dataset
In this implementation, we use the same dataset as the AWF dataset. This dataset contains 100 websites each with 2500 traces. 