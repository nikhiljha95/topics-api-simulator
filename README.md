# Topics API simulator

This repository contains the code to simulate the behaviour of Google's Topics API, together with some examples. The code backs the results shown in the paper **On the Robustness of Topics API to a Re-Identification Attack** by Nikhil Jha, Martino Trevisan, Emilio Leonardi, and Marco Mellia, to be published in PoPETS and to be presented at PETS 2023.

The package offers several possibilities:
- To simulate the behaviour of the Topics API, studying how topics are visited, selected, and exposed.
- To evaluate the probability of a user being $k$-anonymized among the ones that visited the same website.
- To evaluate the probability of a user to be re-identified across two different websites.

The package allows to test behaviour and metrics by changing both environmental parameters (such as the number of users and their visiting rate) and Topics API design parameters, such as the number of $z$ topics which consist in the user profile, or the probability $p$ of the Topics API exposing a random topic instead of a true one.

## Table of contents

1. Content of the repository
2. The code
3. The dataset

## 1. Content of the repository

The repository is organized as such:

- a `topics_api_simulator` module that contains all the file needed to simulate the behaviour of the Topics API in its fundamental aspects;
- an `examples.ipynb` file showing how to use the module;
- the `lambda-user-topic.csv` file, containing the input, real-users-based data. See below for further information.

## 2. The code

The entry point to run the simulation is the `Simulator` class, which in turns instantiates and manages the `Users` and the `Website` classes. The logic behind the code is expressed in the paper:

- Section 2 of the paper explains the notation, and the data structure used;
- Section 4 describes the methods for generating personas from user data;
- Section 5 and 6 detail the used metrics. Please note that the so-called Loose Attack is not included in the PoPETS paper, and will be considered for future work.

While `examples.ipynb` only tests different values of $k$-anonymity, several other parameters can be tuned to test the behaviourof the Topics API:

- `N`: the length of the simulation, in epochs;
- `nusers`: the number of users in the system;
- `T`: the number of topics (tuning this parameters requires a new input dataset for meaningful results);
- `z`: the number of per-epoch top topics that define the user's temporary profile;
- `p`: the rate at which the API exposes a random topic;
- the method chosen to enlarge the dataset. Please refer to the paper for more information.

## 3. The dataset

As input dataset, we offer a post-processed version of the data we collected during the [PIMCity](https://www.pimcity-h2020.eu/) project.

The data contains the average rate of visit for every user on every topic, based on the four-months collection period. Each cell $(i,j)$ contains the rate at which user $i$ has visited topic $j$.

The data contains data about 349 topics from 268 users that visited at least 10 webpages in every epoch - i.e., a week.
