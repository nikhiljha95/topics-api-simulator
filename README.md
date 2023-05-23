# Topics API simulator

This repository contains the code to simulate the behaviour of Google's Topics API, together with some examples. The code backs the results shown in the paper **On the Robustness of Topics API to a Re-Identification Attack** by Nikhil Jha, Martino Trevisan, Emilio Leonardi, and Marco Mellia, to be published in PoPETS and to be presented at PETS 2023.

The package offers several possibilities:
- To simulate the behaviour of the Topics API, studying how topics are visited, selected, and exposed.
- To evaluate the probability of a user being $k$-anonymized among the ones that visited the same website.
- To evaluate the probability of a user to be re-identified across two different websites.

Please refer to the paper for an insight on the notation and the data structures used, together with the logic behind the code.