import numpy as np
from collections import Counter
from scipy.stats import binom

def get_lambdas(df_lambdas, nusers, T, rng, method='original'):
    """Generate the lambdas for a number of users.
    Three methods are available.
    
    The first methods simply uses the lambdas and the number of
    users present in the dataset.
    
    The second method  starts from the marginals of i) topics 
    per user and of i) lambdas per topic to infer the topics
    in which the users is intereted, and the average weekly 
    visits.
    
    The third method consists in performing a dataset 
    augmentation by combining together differnt parent users 
    and create a child user.
    
    Input
    -----
    df (pandas.DataFrame): the user x topic visit rate in an epoch
    nusers (int): number of users in the system
    T (int): number of topics
    rng (numpy.random.Generator): the random number generator
    method (str): the method to generate personas
    
    Output
    ------
    _ (numpy.ndarray): an array contianing the lambda for every persona and topic
    """
    
    ###################################################
    ################# ORIGINAL METHOD #################
    ###################################################
    
    if method=='original':
        
        if nusers > df_lambdas.shape[0]:
            print(f'The required nusers required ({nusers}) is larger than the ' +
                  f'dataset size ({df_lambdas.shape[0]}) and thus will be ignored. '+
                  f'The dataset size will be used instead.')
            original_lambdas = df_lambdas
        else:
            idx = rng.choice(range(df_lambdas.shape[0]), size=nusers, replace=False)
            original_lambdas = df_lambdas[idx,:]
        
        return original_lambdas
        
    ###################################################
    ################### I.I.D. METHOD #################
    ###################################################
    
    elif method=='iid':
        
        # GET DISTRIBUTION OF NUMBER OF TOPICS PER USERS
        
        # DISCLAIMER: I chose to create an histogram of the number of
        # topics visited by users in the original dataframe. Then, for 
        # every user to generate, I sample a bin in the histogram based
        # on its population in the original dataset, then randomly
        # sample a value inside the bin.
        
        active_topics_per_user_df = np.count_nonzero(df_lambdas > 0, axis = 1)
        
        sizes, bins = np.histogram(active_topics_per_user_df)
        
        chosen_bins = rng.choice(range(len(sizes)),
                                 p=sizes/sum(sizes),
                                 size=nusers)
        
        topics_per_user = rng.integers(low=bins[chosen_bins].astype(int),
                                       high=bins[chosen_bins+1].astype(int))
        
        # FILL USER TOPICS
        
        users_per_topic = (df_lambdas>0).sum(axis=0)
        
        #is the following section improvable? Would like to remove the two loops
        user_topics = []
        for x in topics_per_user:
            user_topics.append(rng.choice(len(users_per_topic),
                                          size=x,
                                          p=users_per_topic/sum(users_per_topic),
                                          replace=False))
        
        # ASSIGN LAMBDAS TO CHE CHOSEN TOPICS
        
        # calculate the lambdas for topic, evaluated according to the users
        # who actually visited the topic
        
        #trick to avois divisions by zero (the numerator will be zero anyway)
        users_per_topic[users_per_topic==0] = 1
        
        non_zero_lambda_mean = np.true_divide(df_lambdas.sum(axis=0),
                                              users_per_topic)
        
        
        marginals_lambdas = np.zeros((nusers, len(non_zero_lambda_mean)))    
        for i,topic in enumerate(user_topics):
            marginals_lambdas[i,topic] = non_zero_lambda_mean[topic] #add noise?
            
        return marginals_lambdas
        
    ###################################################
    ################# CROSSOVER METHOD ################
    ###################################################
    
    elif method=='crossover':
        
        fathers = rng.choice(df_lambdas, size=nusers)
        mothers = rng.choice(df_lambdas, size=nusers)
        
        mask = rng.choice(2, size=(nusers, df_lambdas.shape[1]))
        
        combination_lambdas = np.where(mask, fathers, mothers)
        
        mask2 = rng.choice(df_lambdas.shape[1])
        
        return combination_lambdas
        
    else:
        raise ValueError('Unknown method, try either \'original\', '+
                         '\'iid\' or \'crossover\'')
        
        
        
        

def get_kanon_prob(profiles, kanon):
    """Evaluate the ratio of k-anonymized users
    
    Input
    -----
    profiles (list of sets): the list of users' profiles
    kanon (int): the required k-anonymity level
    
    Output
    ------
    _ (float): the ratio of k-anonymized users
    """
    
    profile_counts = Counter([frozenset(x) for x in profiles])
    anonymized = 0
    for v in profile_counts.values():
        if v >= kanon:
            anonymized += v
    return anonymized/sum(profile_counts.values())




def get_threshold(epochs, p, pmin, T, verbose=False):
    """Return the threshold to choose to limit
    the probability of a topic in the profile being a random topic
    
    Input
    -----
    epochs (int): the epoch at which the threshold must be evaluated
    p (float): the random topic probability
    pmin (float): if the probability of having a random topic in the
                  profile exceeds this value, increase the threshold
    T (int): the number of topics
    verbose (bool): whether to show a verbose output
    """
    
    thresh = 1
    binomial = binom(epochs+1, p/T)
    
    #probability of a random topic to appear at least thresh times
    prob_repeat_random_topic = 1 - binomial.cdf(thresh-1)
    
    #increse the threshold if the probability is too high
    while prob_repeat_random_topic > pmin:
        
        if verbose:
            print(f'Probability with epoch={epochs}, p={p}, T={T}, '+
                  f'thresh={thresh} too high ({prob_repeat_random_topic:.2E}), increasing thresh')
        
        thresh += 1
        prob_repeat_random_topic = 1 - binomial.cdf(thresh-1) 
        
    return thresh