import numpy as np
from numpy.random import default_rng
from collections import Counter
import itertools

from .utils import *
from .users import * 
from .website import *

    
class Simulator():
    """Class used to simulate the behaviour of Topics API"""
    
    def __init__(self, df, params, seed=None):
        
        # set the input dataframe
        self.df = df
        # set the parameters
        self.nusers = params["nusers"]
        self.method = params["method"]
        self.z = params["z"]
        self.nepochs = params["nepochs"]
        self.p = params["p"]
        self.taxonomy = params["taxonomy"]
        self.z = params["z"]
        self.pmin = params["pmin"]
        
        # set the default random number generator
        self.rng = default_rng(seed=seed)
        
        # assign the correct number of topics
        if self.taxonomy == 'v1':
            self.T = 349
        elif self.taxonomy == 'v2':
            self.T = 469
        
        # get lambdas based on input parameters
        self.lambdas = get_lambdas(self.df,
                                   self.nusers,
                                   self.T,
                                   self.rng,
                                   method=self.method)
        
        # define users and two websites
        self.users = Users(self.lambdas, self.rng, self.z, self.T)
        self.websites = [
            Website(
                np.array(self.lambdas).shape[0],
                self.T,
                self.p,
                self.pmin,
                0),
            Website(
                np.array(self.lambdas).shape[0],
                self.T,
                self.p,
                self.pmin,
                1)
        ]
        
        
    def simulate_epochs(self):
        """Method used to simulate visits by users and
        queries by websites
        """
        
        for epoch in range(self.nepochs):

            # for every user, simulate her visits...
            self.users.simulate_epoch_visits()
            # ...and compute her profiles
            self.users.create_profiles()
            
            for w in self.websites:
                # simulate each of the websites querying
                # the Topics API                
                w.query_topics(self.users, self.p)

    def _simulate_if_needed(self):
        """Method to run simulation if not already computed"""
        # if self.users.visits has len 0, it means no simulation
        # has been run yet
        if len(self.users.visits) == 0:
            self.simulate_epochs()
    
    
    def get_prob_kanon(self,k=2, denoised=True):
        """Calculate the users' probability of being k-anonymized"""
        
        probs = []
        self._simulate_if_needed()
        
        for epoch in range(self.nepochs):
            # we arbitrarily choose the first of the two websites
            # as an estimator of the k-anonymity probability
            # choosing the second website should lead to similar results
            
            # get the reconstructed profiles at the epoch of interest
            rec_profiles = self.websites[0].get_reconstructed_profiles(epoch, denoised=denoised)
            probs.append(get_kanon_prob(rec_profiles, k))
                              
        return np.array(probs)
            
    
    def is_temporal_coincidence_high(self, u1, u2, exp_w1, exp_w2, thresh, epoch):
        """Check the temporal coincidence filter.
        
        Temporal coincidence means the number of times two users expose the same 
        topics in the same epoch on different websites.
        
        If such number is high, it is more possible that the two users are the
        same user.
        
        Input
        -----
        u1 (int): numerical index of the first user
        u2 (int): numerical index of the second user
        exp_w1 (numpy.ndarray): array describing the topic exposed in epoch i 
                                by user j on website 1
        exp_w2 (numpy.ndarray): array describing the topic exposed in epoch i 
                                by user j on website 2
        thresh (float): the minimum acceptable thresh value
        epoch (int): the epoch taken into consideration
        
        Output
        ------
        _ (boolean): if the temporal coincidence value is larger than threshold
        """
        
        # consider the number of times in which u1 and u2 expose the same topic
        same_exposed_topic = np.equal(exp_w1[:epoch+1,u1], exp_w2[:epoch+1,u2])
        sum_temporal_coincidence = sum(same_exposed_topic)
        
        #return whether such value is above the set threshold
        return sum_temporal_coincidence >= thresh
    
        
    def get_reconstruction_prob(self, denoised=True):
        """Calculate the probability of a user being re-identified,
        across epochs, using a STRICT criteria.
        
        A user is considered as re-identified if the set of topics
        reconstructed on website 1 and website 2 are unique of both 
        websites and identical across websites.
        
        False positives may happen if two unique and matched profiles
        do not belong to the same user.
        
        Input
        -----
        denoised (bool): whether to evaluate the probability of the
                         Denoised Reconstructed Profile (True) or 
                         the Global Reconstructed Profile (False)
                         
        Output
        ------
        tps (list): the rate at which a user is correctly matched
        fps (list): the rate at which a user is incorrectly matched
        """
        
        self._simulate_if_needed()
        
        tps = []
        fps = []
        
        # compute true and false positives for every epoch    
        for epoch in range(self.nepochs):

            # get the reconstructed profiles for both websites
            rw1 = np.array(self.websites[0].get_reconstructed_profiles(epoch, denoised=denoised))
            rw2 = np.array(self.websites[1].get_reconstructed_profiles(epoch, denoised=denoised))

            # users who have the same profile on both the websites
            same_profile = np.equal(rw1, rw2)

            # count occurences of profiles in the same website
            groups1 = Counter([frozenset(x) for x in rw1])
            groups2 = Counter([frozenset(x) for x in rw2])

            # users who are not 2-anonymous
            not_anonymous1 = [groups1[frozenset(x)]<2 for x in rw1]
            not_anonymous2 = [groups2[frozenset(x)]<2 for x in rw2]

            # users which are not 2-anonymous on both websites
            unique_in_both = np.logical_and(not_anonymous1, not_anonymous2)

            # users which are matched across websites
            matched = np.logical_and(unique_in_both, same_profile)

            # user which are unique in the respective websites, but not on both
            unmatched_unique1 = rw1[np.logical_and(not_anonymous1, ~unique_in_both)]
            unmatched_unique2 = rw2[np.logical_and(not_anonymous2, ~unique_in_both)]

            # FALSE POSITIVES: equal to someone else in the set, both unique
            # TPs cannot appear in this list, because they satisfy unique_in_both
            bad_match = sum([uu1==uu2 for uu1, uu2 in itertools.product(unmatched_unique1, unmatched_unique2)])

            tp = matched.mean()
            fp = bad_match / self.nusers

            tps.append(tp)
            fps.append(fp)
            
        
        return tps, fps
    
    def get_reconstruction_prob_loose(self):
        """Calculate the probability of a user being re-identified,
        across epochs, using a LOOSE criteria.
        
        A user is considered as re-identified if all the items included 
        in the *denoised* reconstructed profile on website 1 are also
        included in the *global* reconstructed profile of website 2,
        and viceversa.
        
        To conisder a match, the two *denoised* profiles must be unique
        in each website.
        
        False positives may happen if such condition does not refer to
        different users.
        
        To limit false positives, we perform a temporal coincidence
        analysis. I.e., we count how many times the two users expose
        the same topic in the same epoch. Under a stationary assumption,
        this event should happen a little less than 1/z of the times,
        if we account for the random topic. We thus introduce a minimum
        amount of times the two users must expose the same topic in
        order for them to be included in the TPR/FPR.
        
        
        Output
        ------
        tps (list): the rate at which a user is correctly matched
        fps (list): the rate at which a user is incorrectly matched
        """
        
        self._simulate_if_needed()
        
        # get the reconstructed profiles for both websites
        exp_topics_history_w1 = self.websites[0].get_exposed_topics_history()
        exp_topics_history_w2 = self.websites[1].get_exposed_topics_history()
        
        tps = []
        fps = []
        
        # compute TPR and FPR for every epoch
        for epoch in range(self.nepochs):
            
            # the minimum number of time two users must expose the same topic
            # on different website. We set this thresh at half of the
            # average number of times this event happens at the reference
            # epoch
            thresh = epoch/self.z/2

            # get denoised and global reconstructed profiles
            gw1 = self.websites[0].get_reconstructed_profiles(epoch, denoised = False)
            rw1 = self.websites[0].get_reconstructed_profiles(epoch)
            gw2 = self.websites[1].get_reconstructed_profiles(epoch, denoised = False)
            rw2 = self.websites[1].get_reconstructed_profiles(epoch)

            # count occurences of profiles in same website
            groups1 = Counter([frozenset(x) for x in rw1])
            groups2 = Counter([frozenset(x) for x in rw2])

            # users who are not 2-anonymous
            not_anonymous1 = [groups1[frozenset(x)]<2 for x in rw1]
            not_anonymous2 = [groups2[frozenset(x)]<2 for x in rw2]

            # creating a list of tuples, each composed as such:
            # 0: user id
            # 1: reconstructed profile
            # 2: denoised reconstructed user profile
            # one entry for every user which is not anonymous in the website
            unique_profiles1 = [(idx,gw,rw) for idx,(gw,rw,na) in enumerate(zip(gw1, rw1, not_anonymous1)) if na]
            unique_profiles2 = [(idx,gw,rw) for idx,(gw,rw,na) in enumerate(zip(gw2, rw2, not_anonymous2)) if na]
            
            # compare non-anonymous users for the two websites
            
            # define true positive and false positive counters
            tp = 0
            fp = 0
            
            for up1 in unique_profiles1:
                matches = 0
                tp_temp = 0
                fp_temp = 0
                
                for up2 in unique_profiles2:
        
                    # check if the denoised reconstructed profiles are contained 
                    # in the other's global reconstructed profile, and viceversa
                    if up1[2].issubset(up2[1]) and up2[2].issubset(up1[1]):
                        
                        # continue only if the two users have a sufficient 
                        # high number of coincident topics
                        if  self.is_temporal_coincidence_high(up1[0],
                                                              up2[0],
                                                              exp_topics_history_w1,
                                                              exp_topics_history_w2,
                                                              thresh,
                                                              epoch):

                            #count the number of matches for each up1
                            matches += 1

                            # if the two indexes (hence, users) correspond, it's a true positive
                            if up1[0]==up2[0]:
                                tp_temp += 1

                            # else, it's a false positive
                            else:
                                fp_temp += 1

                # if up1 counts more than 1 match, it's not possible to 
                # re-identify - hence, don't count
                if matches <= 1:
                    
                    tp += tp_temp
                    fp += fp_temp


            tps.append(tp/self.nusers)
            fps.append(fp/self.nusers)

        return tps, fps
    
    
    def get_reconstruction_prob_loose_fixed_epoch(self, epoch):
        """Calculate the probability of a user being re-identified,
        across epochs, using a LOOSE criteria.
        
        Same behaviour of get_reconstruction_prob_loose, except
        the returned values refer to a fixed epoch, while varying
        the temporal coincidence threshold.
        
        Please refer to get_reconstruction_prob_loose for details
        on the matching criteria and the temporal coincidence
        threshold.        
        
        Output
        ------
        tps (list): the rate at which a user is correctly matched
        fps (list): the rate at which a user is incorrectly matched
        """
        
        self._simulate_if_needed()
        
        # get the reconstructed profiles for both websites
        exp_topics_history_w1 = self.websites[0].get_exposed_topics_history()
        exp_topics_history_w2 = self.websites[1].get_exposed_topics_history()
        
        # get denoised and global reconstructed profiles
        gw1 = self.websites[0].get_reconstructed_profiles(epoch, denoised = False)
        rw1 = self.websites[0].get_reconstructed_profiles(epoch)
        gw2 = self.websites[1].get_reconstructed_profiles(epoch, denoised = False)
        rw2 = self.websites[1].get_reconstructed_profiles(epoch)

        # count occurences of profiles in same website
        groups1 = Counter([frozenset(x) for x in rw1])
        groups2 = Counter([frozenset(x) for x in rw2])

        # users who are not 2-anonymous
        not_anonymous1 = [groups1[frozenset(x)]<2 for x in rw1]
        not_anonymous2 = [groups2[frozenset(x)]<2 for x in rw2]

        # creating a list of tuples, each composed as such:
        # 0: user id
        # 1: reconstructed profile
        # 2: denoised reconstructed user profile
        # one entry for every user which is not anonymous in the website
        unique_profiles1 = [(idx,gw,rw) for idx,(gw,rw,na) in enumerate(zip(gw1, rw1, not_anonymous1)) if na]
        unique_profiles2 = [(idx,gw,rw) for idx,(gw,rw,na) in enumerate(zip(gw2, rw2, not_anonymous2)) if na]
        
        tps = []
        fps = []                       
            
        for thresh in range(epoch):
                            
            # define true positive and false positive counters
            tp = 0
            fp = 0
            
            for up1 in unique_profiles1:
                matches = 0
                tp_temp = 0
                fp_temp = 0
                
                for up2 in unique_profiles2:
                    
                    # check if the denoised reconstructed profiles are 
                    # contained in the other's global reconstructed profile
                    if up1[2].issubset(up2[1]) and up2[2].issubset(up1[1]):
                        
                        #count the number of matches for each up1
                        matches += 1

                        # if the two indexes (hence, users) correspond, 
                        # it's a true positive
                        if up1[0]==up2[0]:
                            tp_temp += 1

                        # else, it's a false positive
                        else:
                            fp_temp += 1

                # if up1 counts more than 1 match, it's not possible to 
                # re-identify - hence, don't count
                if matches == 1:
                    
                    tp += tp_temp
                    fp += fp_temp


            tps.append(tp/self.nusers)
            fps.append(fp/self.nusers)

        return tps, fps
    
    
    def unweighted_hamming_attack(self):
        
        self._simulate_if_needed()
        
        # get the reconstructed profiles for both websites
        exp_topics_history_w1 = self.websites[0].get_exposed_topics_history()
        exp_topics_history_w2 = self.websites[1].get_exposed_topics_history()
        
        tps = []
        fps = []
        
        for epoch in range(self.nepochs):
            tp = 0
            fp = 0
            test_users = min([self.nusers, 1000])
            for up1 in range(exp_topics_history_w1.shape[1])[:test_users]:
                
                most_similar_value= -1
                most_similar_candidates = []
                
                for up2 in range(exp_topics_history_w2.shape[1]):
                    
                    # consider the number of times in which u1 and u2 expose the same topic
                    same_exposed_topic = np.equal(exp_topics_history_w1[:epoch+1,up1], exp_topics_history_w2[:epoch+1,up2])
                    sum_temporal_coincidence = sum(same_exposed_topic)
                    
                    if sum_temporal_coincidence >= most_similar_value:
                        most_similar_value = sum_temporal_coincidence
                        most_similar_candidates.append(up2)
                        
                if up1 == self.rng.choice(most_similar_candidates):
                    tp += 1
                else:
                    fp += 1

            tps.append(tp/test_users)
            fps.append(fp/test_users)

        return tps, fps
    
    
    def asymmetric_weighted_hamming_attack(self):

        self._simulate_if_needed()
        
        # get the reconstructed profiles for both websites
        exp_topics_history_w1 = self.websites[0].get_exposed_topics_history()
        exp_topics_history_w2 = self.websites[1].get_exposed_topics_history()
        
        topics, counts = np.unique(exp_topics_history_w1, return_counts=True)
        topic_count_dict = {int(k):v for k,v in zip(topics, counts)}
        topic_count_w1 = np.array([topic_count_dict.get(k,0) for k in range(self.T)])
        
        qin = (1-self.p) / self.z + self.p / self.T
        qout = self.p / self.T
        
        topic_top_k_prob = qin * topic_count_w1 / self.nepochs / self.nusers / (qin-qout)
        #topic_top_k_prob = (topic_count_w2 - (qout * self.nepochs * self.nusers)) / self.nepochs / self.nusers / (qin-qout)
        
        num = qout+ (qin - qout)* qin* topic_top_k_prob
        den = (qout + (qin - qout) * topic_top_k_prob)
        
        match_weight = -np.log(num/den)
        
        mismatch_weight = -np.log(
            qout
            + (qin - qout)
            * (self.z - 1)
            * topic_top_k_prob
            / (self.z - topic_top_k_prob)
        )
        
        tps = []
        fps = []
        
        for epoch in range(self.nepochs):
            tp = 0
            fp = 0
            
            test_users = min([self.nusers, 1000])
            for up1 in range(exp_topics_history_w1.shape[1])[:test_users]:
                
                min_distance = np.inf
                best_candidates = []
                
                topic_seq_match_weight = match_weight[exp_topics_history_w1[:epoch+1,up1]]
                topic_seq_mismatch_weight = mismatch_weight[exp_topics_history_w1[:epoch+1,up1]]
                
                for up2 in range(exp_topics_history_w2.shape[1]):
                    
                    same_exp_topic = np.equal(exp_topics_history_w1[:epoch+1,up1], exp_topics_history_w2[:epoch+1,up2])
                    
                    distance = np.sum(np.where(same_exp_topic, topic_seq_match_weight, topic_seq_mismatch_weight))
                    
                    if distance <= min_distance:
                        min_distance = distance
                        best_candidates.append(up2)
                        
                if up1 == self.rng.choice(best_candidates):
                    tp += 1
                else:
                    fp += 1

            tps.append(tp/test_users)
            fps.append(fp/test_users)

        return tps, fps
    
    
    
    
    def asymmetric_weighted_hamming_attack_v2(self):

        self._simulate_if_needed()
        
        # get the reconstructed profiles for both websites
        exp_topics_history_w1 = self.websites[0].get_exposed_topics_history()
        exp_topics_history_w2 = self.websites[1].get_exposed_topics_history()
        
        topics, counts = np.unique(exp_topics_history_w1, return_counts=True)
        topic_count_dict = {int(k):v for k,v in zip(topics, counts)}
        topic_count_w1 = np.array([topic_count_dict.get(k,0) for k in range(self.T)])
        
        qin = (1-self.p) / self.z + self.p / self.T
        qout = self.p / self.T
        
        #topic_top_k_prob = qin * topic_count_w2 / self.nepochs / self.nusers / (qin-qout)
        topic_top_k_prob = (topic_count_w1 - (qout * self.nepochs * self.nusers)) / self.nepochs / self.nusers / (qin-qout)
        
        num = qout+ (qin - qout)* qin* topic_top_k_prob
        den = (qout + (qin - qout) * topic_top_k_prob)
        
        match_weight = -np.log(num/den)
        
        mismatch_weight = -np.log(
            qout
            + (qin - qout)
            * (self.z - 1)
            * topic_top_k_prob
            / (self.z - topic_top_k_prob)
        )
        
        tps = []
        fps = []
        
        for epoch in range(self.nepochs):
            tp = 0
            fp = 0
            
            test_users = min([self.nusers, 1000])
            for up1 in range(exp_topics_history_w1.shape[1])[:test_users]:
                
                min_distance = np.inf
                best_candidates = []
                
                topic_seq_match_weight = match_weight[exp_topics_history_w1[:epoch+1,up1]]
                topic_seq_mismatch_weight = mismatch_weight[exp_topics_history_w1[:epoch+1,up1]]
                
                for up2 in range(exp_topics_history_w2.shape[1]):
                    
                    same_exp_topic = np.equal(exp_topics_history_w1[:epoch+1,up1], exp_topics_history_w2[:epoch+1,up2])
                    
                    distance = np.sum(np.where(same_exp_topic, topic_seq_match_weight, topic_seq_mismatch_weight))
                    
                    if distance <= min_distance:
                        min_distance = distance
                        best_candidates.append(up2)
                        
                if up1 == self.rng.choice(best_candidates):
                    tp += 1
                else:
                    fp += 1

            tps.append(tp/test_users)
            fps.append(fp/test_users)

        return tps, fps