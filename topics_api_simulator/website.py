import numpy as np
from collections import Counter
import copy
from .utils import * 

class Website():
    """Class used to simulate the behaviour of a website in the
    Topics API framework
    """
    
    def __init__(self, nusers, T, p, pmin, wid):
        self.nusers = nusers
        self.T = T
        # count of times each user has exposed each topic
        self.topics_counts_per_user = np.array([[[0] * T] * nusers])
        # parameters to calculate the denoising threshold
        self.p = p
        self.pmin = pmin
        self.wid = wid
    
    
    def query_topics(self, users, p):
        """Get the per-epoch topics offered by the users and
        update the data structures
        
        Input
        -----
        users (topics_api_simulator.users): set of users to query
        p (float): expected rate of random topics
        
        Output
        ------
        last_topics_counts_per_user (numpy.ndarray): a 2D array 
                                                     containing the 
                                                     updated count of
                                                     visits by each user
                                                     on each topic
        """
        
        # query the topics
        topics = users.expose_topics(p,self.wid)
        # update the user-topic counts
        last_topics_counts_per_user = copy.deepcopy(self.topics_counts_per_user[-1])   
        last_topics_counts_per_user[range(len(topics)), topics] += 1
        
        # append the last topic into the topics observed by each user
        self.topics_counts_per_user = np.concatenate(
            (
                self.topics_counts_per_user,
                np.expand_dims(
                    last_topics_counts_per_user,
                    axis=0
                )
            )
        )
        
        return last_topics_counts_per_user
    
    def get_reconstructed_profiles(self, epoch, denoised=True):
        """Get Reconstructed Profiles
        
        Input
        -----
        epoch (int): the epoch at which profiles are evaluated
        denoised (boolean): whether to apply the denoising algorithm, or not
        
        Output
        ------
        users_profile (list of sets): the profile for each user
        """
        
        # initialize users profiles
        users_profile = [set() for _ in range(self.nusers)]
        # choose the trehsold, 1 if denoising not required
        if denoised:
            thresh = get_threshold(epoch, self.p, self.pmin, self.T)
        else:
            thresh = 1
        #for every topic above threshold, add it to the user profile
        profile_index = np.argwhere(self.topics_counts_per_user[epoch] >= thresh)
        for user, topic in profile_index:
            users_profile[user].add(topic)
        return users_profile
    
    def get_exposed_topics_history(self):
        """Return the exposed topics"""
        # subtract the count from the previous one
        diff = self.topics_counts_per_user - np.roll(self.topics_counts_per_user, axis=0, shift=1)
        # found user and topics where the difference is larger than 0
        epoch_users_topics = np.array(np.where(diff>0))
        # epoch_users_topics index for epochs has an offset of 1
        epoch_users_topics[0] = epoch_users_topics[0] - 1
        # fill a structure containing the topic exposed by users in every epoch
        exp_topics = np.empty((diff.shape[0]-1, diff.shape[1]))
        exp_topics[epoch_users_topics[0], epoch_users_topics[1]] = epoch_users_topics[2]
        return exp_topics