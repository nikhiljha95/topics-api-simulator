import numpy as np
import copy

class Users():
    """Class used to simulate the behaviour of a set of users
    according to the Topics API"""
    
    def __init__(self, lambdas, rng, z=5, T=349):
        # the input average visiting rate, for every (user,topic) couple
        self.lambdas = np.array(lambdas)
        self.z = z
        self.nusers = self.lambdas.shape[0]
        self.T = T
        self.rng = rng
        # data structures to be filled
        self.visits = []
        self.profiles = []
        self.exposed_topics = dict()
        self.masks = dict()
        
    def simulate_epoch_visits(self):
        """Simulate the number of times each user visits each topic,
        in an epoch
        
        Output
        ------
        visits (numpy.ndarray): a user x topic array indicating the
                                number of simulated visits, plus a
                                random <1 value to break ties
        """
                                                  
        visits = self.rng.poisson(self.lambdas)
        # add random <1 noise to randomize ties
        # since the noise is <1, it does not impact the subsequent ordering
        visits = visits + self.rng.random(self.lambdas.shape)
        self.visits.append(visits)
        return visits
    
    def create_profiles(self): 
        """Calculate the top z topics for every user
        
        Output
        ------
        profiles (numpy.ndarray): a user x z array indicating the top z
                                  topics per each user
        """
        
        profiles = np.argpartition(self.visits[-1], -self.z, axis = 1)[:,-self.z:]
        self.profiles.append(profiles)
        return profiles
    
    def expose_topics(self, p, wid):        
        """Return a topic from the profile of each user (choosen at random).
        In p of the cases, return a random topic
        
        Input
        -----
        p (float): the average ratio of random topic
        
        Output
        ------
        exposed_topics (numpy.ndarray): a list-like array including the topics
                                        exposed by users
        """
        
        # permute the topics and take the first in the column for every user
        exposed_topics = self.rng.permuted(self.profiles[-1], axis=1)[:,0]
        
        # randomly select the topics that COULD BE used as random replacement
        # one topic per user
        random_topics = self.rng.choice(range(self.T), self.profiles[-1].shape[0])
        
        # create a mask with a ratio of True equal to p
        mask = self.rng.choice([True, False], p=[p, 1.-p], size=self.profiles[-1].shape[0])
        
        # replace the true topic with a random one according to the mask
        exposed_topics[mask] = random_topics[mask]
        
        #update the internal data structures
        if wid in self.exposed_topics:
            self.exposed_topics[wid] = np.append(
                self.exposed_topics[wid],
                exposed_topics.reshape(1,-1),
                axis=0
            )
        else:
            self.exposed_topics[wid] = copy.deepcopy(exposed_topics).reshape(1,-1)
        if wid in self.masks:
            self.masks[wid] = np.append(
                self.masks[wid],
                mask.reshape(1,-1),
                axis=0
            )
        else:
            self.masks[wid] = copy.deepcopy(mask).reshape(1,-1)

        return exposed_topics
    
    def get_visits(self):
        """Return the visits"""
        return np.array(self.visits)
    
    def get_profiles(self):
        """Return the profiles"""
        return np.array(self.profiles)
    
    def get_exposed_topics(self):
        """Return the exposed topics"""
        return self.exposed_topics
    
    def get_masks(self):
        """Return the masks"""
        return self.masks