"""
fuzzy_mining.py: Contains basic implementation of some fuzzy mining algorithms.

@input: A dataframe contiaing your data.
@output: A fuzzy rule base extracted from the input.

@author: Fanqing Xu
@reference: 
    Main ref. [The WM Method Completed: A Flexible Fuzzy System Approach 
to Data Mining, Wang, 2003]
    Main package used is 'scikit-fuzzy' which is a prerequisite.
    [https://github.com/scikit-fuzzy/scikit-fuzzy]
"""
import numpy as np
import skfuzzy as fuzz
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skfuzzy import control as ctrl

import time
import warnings

class FuzzyObj:
    """
    Basic object containing fuzzy rule base and fuzzy relations.

    Parameters
    ----------
    dataset: pandas dataframe
        Dataset containing training dataset. Must be a pandas dataframe for now,
        with each raw representing a (x_1, x_2, ..., x_n, y). Only one y is supported.
    names: string list
        Names for all Antecedent and Consequent. If not specified, column names of
        dataset will be used. There might be unpredictable error if the column names
        are complicated. Recommended.
    
    
    Methods
    -------
    to be specified
    
    """

    def __init__(self, dataset, names='default'):
        """
        Initialize fuzzy object.

        Parameters
        ----------
        dataset: pandas dataframe
            Must be a pandas df for now. Each row is a data point.
        names: string list
            Simplified names for each Antecedent and Consequent.
            Easy, short, meaningful names are recommended.
        """
        self.dataset = dataset
        
        if names == 'default':
            self.names = dataset.columns.to_list()
        else:
            self.names = names
        
        self.rule_base = []
        self.extra_rules = []
        self.region_names = []
        self.ctrl = None
        self.group = {}
        self.dict_obj = {}
        self.list_obj = []


    def __repr__(self):
        return "Dataset preview: {0}\n Names:{1}".format(self.dataset, self.names)
        
    def create_fuzzy_variables(self):
        """
        Create fuzzy variables for all Antecedents and Consequent.
        """
        # use min and max of all data to define the total interval of features
        # -1 and +1 to avoid edge cases
        INCRE = 0.1
        minimum = np.array(self.dataset.min().to_list()) - 1
        maximum = np.array(self.dataset.max().to_list()) + 1

        dict_obj = {}
        for n in range(len(self.names) - 1):
            dict_obj[self.names[n]] = ctrl.Antecedent(np.arange(minimum[n], maximum[n], INCRE), self.names[n])

        dict_obj[self.names[-1]] =  ctrl.Consequent(np.arange(minimum[-1], maximum[-1], INCRE), self.names[-1])

        self.dict_obj = dict_obj
        self.list_obj = [v for _, v in dict_obj.items()]
        print("Fuzzy variables are created with {} Antecedents and {} Consequent:\n{}"
              .format(len(self.list_obj)-1, 1, self.list_obj))

    def auto_fuzzify(self, names='default', n_regions=7):
        """
        Fuzzify the fuzzy variables into fuzzy regions using triangular membership functions.
        Another method for users to customize the shape of membership function will be added.
        
        Parameter
        ---------
        n_regions: int
            Number of regions you want to define. Default is 7.
        names: string list
            Names for each regions. e.g. "Very cold", "Cold", ..., "Hot", "Very hot"
            len(names) must matches n_regions.
            If you pass in names, n_regions will be automatically set.
            If you use default naming convention, you can still specify n_regions.
            Only support same names for each region for now.
        
        
        """
        if names == 'default':
            names = [str(x+1) for x in range(n_regions)]
        else:
            n_regions = len(names)
        
        self.region_names = names
        
        for obj in self.list_obj:
            obj.automf(names=names)
        
        print("Fuzzification is done. View membership function using .view(name, region):\nRegions are: {}"
              .format(names))
        print("Preview of Consequent y:\n")
        
        self.list_obj[-1].view()
        
    def view(self, name, region='none'):
        """
        View the membership function of any Antecedent or Consequent (specified by name).
        And you can specify the region you want to highlight (specified by region).
        
        Parameter
        ---------
        name: string
            Name of the Antecedent or Consequent you want to view.
        region: string
            Name of the region you want to view.
        """
        if region == 'none':
            self.dict_obj[name].view()
        else:
            self.dict_obj[name][region].view()
            
            
    def extract(self):
        """
        Extract fuzzy rule base from training data given.
        
        Parameter
        ---------
        
        """
        X_train = self.dataset
        train_idx = X_train.index
        ant_con = self.list_obj
        names = self.region_names
        
        temp_rule_base = [] # temporary rule_base for grouping, every rule in it is unique
        groups = {} # initial group, structure: {'index': [[data1],[data2]...], ...}

        # Loop over all training data pairs
        for it in train_idx:
            # x stands for a row in dataframe, a pair of {(xi, Y)}
            x = X_train.loc[it,:].tolist()
            region = [] # region for each feature and Y
            degree = [] # degree = max(membership value(x_i)) for each x component
            # loop over each x component, Y is still needed here
            for i in range(len(x)):
                temp = [] # containing degree for each region of x_i
                # loop over all regions and compute membership function, storing to temp for comparison
                for n in names:
                    # here .interp_membership() have to pass in range of x and membership function of x, both 1d array
                    # cement['CE'] will extract .term.Term object and use .mf to get membership function results
                    temp.append(fuzz.interp_membership(ant_con[i].universe, ant_con[i][n].mf, x[i],zero_outside_x=False))
                # take the region with max degree
                region.append(names[np.argmax(temp)])
                # assign the maximum degree for x_i
                degree.append(np.max(temp))
            # compute weight of rule i by multiplying every degree of x components, Y is still needed now
            rule_weight = np.prod(degree)
            # if the data pair has no weight(degree), means no information can be extracted
            # this step needs further consideration
        #     if rule_weight <= 0:
        #         print(it,region, degree)
        #         continue
            # assign y_i=x[-1], weight and index of data to data_i
            region.extend([x[-1], rule_weight])
            # now region is a list [region of x1, region of x2, ..., region of xn, region of y, value of y, degree/weight]
            # representing training_data_i
            
                
            # group the rules by same IF part (including no conflicting case, which will be filter out later)
            # empty temp rule_base case, initialize rule base using first data pair
            if len(temp_rule_base) == 0:
                # initial group 0, group_id is indexed by the index of temp_rule_base
                temp_rule_base.append(region[:]) 
                # first group ever added
                groups[0] = [region[:]] 
            else:
                # initial flag variable to False
                duplicate = False
                # loop over temp_rule_base inversivly
                for j in temp_rule_base[::-1]:
                    # check for rules with same IF part, last 3 components are region of y, y_i, weight
                    if j[:-3] == region[:-3]:
                        # group it into the same group as j (group_index)
                        # check if has this group, initialize a empty group(list)
                        index = temp_rule_base.index(j)
                        groups[index].append(region)
                        duplicate = True
                        break
                        
                if not duplicate:
                    # add a new unique IF part into temp_rule_base
                    temp_rule_base.append(region)
                    index = temp_rule_base.index(region)
                    groups[index] = [region]
                
        print("{} unique rules(groups) are found.".format(len(groups)))
        # now the groups has all the grouped data  
        self.group = groups

    def resolve_conflict(self, method='majority'):
        """
        Resolve conflict rules. Conflict rules have same IF part but different THEN part,
        you have to select one method to choose from those rules.

        Parameter
        ---------
        method: string
            Method used to resolve conflict. Accepted arguments are
            * 'random' : Randomly select one Consequent y region.
            * 'majority' : Select by majority vote for y region.
            * 'ranked' : Select by weight(degree) of the rules.
            * 'WMcomplete' : Define new y region according to [Wang, 2003]
            * 'semiavg' : Average over integer represented y regions, only supported
                if y regions are e.g. ['1', '2', '3'...].
            If tie, randomly select one.
        """
        rule_base = [] # initialize rule_base

        if method == 'random':
            self.random_select(rule_base)
        elif method == 'majority':
            self.majority_select(rule_base)
        elif method =='ranked':
            self.ranked_select(rule_base)
        elif method == 'WM-complete':
            self.WM_select(rule_base)
        elif method == 'semiavg':
            self.semiavg_select(rule_base)

    def interp_rules(self, extra_data, method='Jaccard'):
        """
        Interpolate rule base for extra data.
        Due to sparsity of the rule base, we here only interpolate rules
        for data needed, e.g. test dataset.
        If the extra dataset find no rules in rule base, this will create a new rule.

        We use nearest neighbor to define the new rule's y region.

        Parameter
        ---------
        extra_data: pandas dataframe
            The dataset you want to find rules. This should be a pandas dataframe like training dataset.
        method: string
            Metric for distance. Accepted arguments are:
            * 'Jaccard' : Using Jaccard similarity as metric.
            * 'plain' : Using a rough way to measure distance. Only supported if region names are
            same list of consecutive integers. e.g. ['1', '2', '3', ...]
            More to be added.
        """
        # extract index of extra data
        extra_idx = extra_data.index
        extra_rules = []
        ant_con = self.list_obj
        names = self.region_names
        rule_base = self.rule_base
        # loop over all test data
        for it in extra_idx:
            # x stands for a row in dataframe, a pair of {(xi, Y)}
            x = extra_data.loc[it,:].tolist()
            region = [] # region for each feature and Y
            # loop over each x component, Y is not needed here, which is our target
            for i in range(len(x)-1):
                temp = [] # containing degree for each region of x_i
                # loop over all regions and compute membership function, storing to temp for comparison
                for n in names:
                    # here .interp_membership() have to pass in range of x and membership function of x, both 1d array
                    # cement['CE'] will extract .term.Term object and use .mf to get membership function results
                    temp.append(fuzz.interp_membership(ant_con[i].universe, ant_con[i][n].mf, x[i]))
                # take the region with max degree
                region.append(names[np.argmax(temp)])
            # now region is a list [region of x1, region of x2, ..., region of xn]
            # representing extra_data_i's IF part

            # Check if it has matching rule
            match = False
            for r in rule_base:
                if region[:] == r[:-1]:
                    # find matches
                    match = True
                    break
            
            # need interpolate the rule_base if not match
            if not match:
                metric = method
                if metric == "Jaccard":
                    # given new rule region[:](IF part), you have to compare its IF part with all rules in the rule_base r[:-2]
                    # the package provide a method to compute similarity of two membership function, .fuzzy_similarity is not 
                    # valid. Using new defined Jaccard Similarity
                    similarity = []
                    for r in rule_base:
                        sim = 0
                        for i in range(len(region)):
                            a = ant_con[i][region[i]].mf
                            b = ant_con[i][r[i]].mf
                            sim += np.trapz(np.fmin(a, b)) / np.trapz(np.fmax(a, b))
                        similarity.append(sim)
                    neighbors_idx = np.where(similarity == np.max(similarity))[0]
                    # check if tie
                    if len(neighbors_idx) == 1:
                        # only one nearest neighbor, use it's y_region directly
                        region.append(rule_base[int(neighbors_idx[0])][-1])
                    else:
                        # 1. average over all tied neighbors
                        # 2. or you randomly select one neighbor, here just randomly select one
                        # use 2. here
                        region.append(rule_base[np.random.choice(neighbors_idx)][-1])
                else:
                    distances = [] # record distances between rule in the rule_base and test_data
                    for r in rule_base:
                        dis = 0
                        p = len(r) - 1 # subtract 1 for y region
                        for i in range(p):
                            dis += np.abs(int(r[i]) - int(region[i]))
                        distances.append(dis/p)
                    # we only try k=1 here, namely one nearest neighbor
                    neighbors_idx = np.where(distances == np.min(distances))[0]
                    # check if tie
                    if len(neighbors_idx) == 1:
                        # only one nearest neighbor, use it's y_region directly
                        region.append(rule_base[int(neighbors_idx[0])][-1])
                    else:
                        # 1. average over all tied neighbors
                        # 2. or you randomly select one neighbor, here just randomly select one
                        # use 2. here
                        region.append(rule_base[np.random.choice(neighbors_idx)][-1])
                # add extra rules(region)
                if it == 395:
                    print(region)
                extra_rules.append(region)
        print("{} extra rules are added after interpolation.".format(len(extra_rules)))

        self.extra_rules.extend(extra_rules)

    def build_rules(self, rule_base=[]):
        """
        Build the fuzzy rules for prediction.

        Parameter
        ---------
        rule_base: list of string list
            Any rules you want to add. Must be the same format with rule_base.
            You can check the rule_base by calling self.rule_base.

        Return
        ------
        rules: list of Rule objects
            Rule objects that will be used in building fuzzy system.
        """
        # combine all rules
        final_rule_base = []
        final_rule_base.extend(self.rule_base)
        if len(rule_base) > 0:
            final_rule_base.extend(rule_base)
        if len(self.extra_rules) > 0:
            final_rule_base.extend(self.extra_rules)

        # .Rule() method also accepts Term
        rules = []
        ant_con = self.list_obj
        for r in final_rule_base:
            rules.append(ctrl.Rule(ant_con[0][r[0]] & ant_con[1][r[1]] & ant_con[2][r[2]]
                        & ant_con[3][r[3]] & ant_con[4][r[4]] & ant_con[5][r[5]] 
                        & ant_con[6][r[6]] & ant_con[7][r[7]], ant_con[-1][r[-1]]))

        # you can check a visualization of rule by .view()
        rules[0].view()

    def build_system(self, rules):
        """
        Build the fuzzy system.

        Parameter
        ---------
        rules: list of Rule objects
            rules built by self.build_rules
        """
        start = time.process_time()
        print("This will take a fairly long time, caused by graph creating process which is used in scikit-fuzzy package.\n")
        # initialize a controlSystem using rules obtained above
        concrete_ctrl = ctrl.ControlSystem(rules)
        # initialize a controlSystemSimulation using controlSystem
        concrete_sys = ctrl.ControlSystemSimulation(concrete_ctrl)
        print("Done. Time used to build fuzzy system: {} sec.".format(time.process_time()-start))
        
        self.ctrl = concrete_sys

    def test(self):
        t_id = np.random.choice(self.dataset.index)
        ant_con = self.list_obj
        concrete = self.ctrl
        names = self.names

        x = self.dataset.loc[t_id,:].tolist()
        for i in range(len(ant_con)-1):
            concrete.input[ant_con[i].label] = x[i]
        # compute Y
        concrete.compute()

        # visualize the output
        ant_con[-1].view(sim=concrete)
        print("predicted y: {}".format(concrete.output[names[-1]]))
        print("true y: {}".format(x[-1]))
        print("error: " + str(concrete.output[names[-1]] - x[-1]))

    def compute_error(self, data, method='R2'):
        """
        Compute training/test error using method.

        Parameter
        ---------
        data: pandas dataframe
            Dataset you want to compute error on. Must be pandas dataframe
        method: string
            Metric used for evaluating rules. Accepted arguments are:
            * 'R2': R^2 score.

        Retrun
        ------
        train_error: float
            training error computed by method.
        """
        # Compute error
        u = []
        v = []
        idx = data.index
        ant_con = self.list_obj
        concrete = self.ctrl

        for t in idx:
            x = data.loc[t,:].tolist()
            # assign inputs
            for i in range(len(ant_con)-1):
                concrete.input[ant_con[i].label] = x[i]
            # compute Y
            try:
                concrete.compute()
                # here we compute R^2 = 1 - u/v, u = ((y_true - y_pred) ** 2).sum(), 
                # v =  ((y_true - y_true.mean()) ** 2).sum()
                # the best score R^2 will be 1.
                y_pred = concrete.output[ant_con[-1].label]
                y_true = x[-1]
                u.append((y_true - y_pred) ** 2)
                v.append(y_true)
            except Exception as e:
                print(e)
                print("no rule for {}".format(t))
        #         u.append((x[-1] - 0)**2)
        #         v.append(x[-1])
                continue    
            
        error = 1 - np.sum(u) / np.sum((v - np.mean(v))**2)
        return error

    def random_select(self, rule_base):
        """
        If conflict, select a y region randomly.

        Parameter
        ---------
        rule_base: list
            A empty list for final rule base.
        """
        groups = self.group

        # loop over each group
        for key, val in groups.items():
                
            # check if only one rule in the group
            if len(val) == 1:
                # add it to rule_base directly
                # val is a list [x_regions, y_region, y_value, degree]
                region = val[0]
                rule_base.append(region[:-2])
            else:
                # randomly select one
                new_rule = val[np.random.choice(range(len(val)))]
                # add rule into rule_base
                rule_base.append(new_rule[:-2])
        print("{} final rules are built.".format(len(rule_base)))
        
        self.rule_base = rule_base


    def majority_select(self, rule_base):
        """
        If conflict, select a y region by majority vote.

        Parameter
        ---------
        rule_base: list
            A empty list for final rule base.
        """
        groups = self.group

        # loop over each group
        for key, val in groups.items():
                
            # check if only one rule in the group
            if len(val) == 1:
                # add it to rule_base directly
                # val is a list [x_regions, y_region, y_value, degree]
                region = val[0]
                rule_base.append(region[:-2])
            else:
                # make a vote
                y_s = []
                for v in val:
                    y_s.append(v[-3])
                _, unique_counts = np.unique(y_s, return_counts=True)
                y_id = np.random.choice(np.where(unique_counts==unique_counts.max())[0],1)[0]
                
                # add rule into rule_base
                new_rule = val[0][:-3] + [y_s[y_id]]
                rule_base.append(new_rule)
        print("{} final rules are built.".format(len(rule_base)))

        self.rule_base = rule_base

    def ranked_select(self, rule_base):
        """
        If conflict, select a y region by weight of rules.

        Parameter
        ---------
        rule_base: list
            A empty list for final rule base.
        """
        groups = self.group

        # loop over each group
        for key, val in groups.items():
                
            # check if only one rule in the group
            if len(val) == 1:
                # add it to rule_base directly
                # val is a list [x_regions, y_region, y_value, degree]
                region = val[0]
                y_region = region[-2]
                rule_base.append(region[:-2])
            else:
                # rank by weight, select region with largest weight
                y_s = []
                for v in val:
                    y_s.append(v[-1])
                y_id = np.where(y_s == np.max(y_s))
                
                # add rule into rule_base
                new_rule = val[0][:-3] + [val[y_id][-3]]
                rule_base.append(new_rule)
        print("{} final rules are built.".format(len(rule_base)))
        
        self.rule_base = rule_base


    def WM_select(self, rule_base):
        """
        If conflict, create new y regions according to [Wang, 2003].
        Create the new y region using triangular membership function with parameters
        [y_c - sigma, y_c, y_c + sigma], where y_c is the weighted average of y value,
        sigma is the variance.
        The new region is named default "region + index"

        Parameter
        ---------
        rule_base: list
            A empty list for final rule base.
        """
        groups = self.group

        # loop over each group
        for key, val in groups.items():
                
            # check if only one rule in the group
            if len(val) == 1:
                # add it to rule_base directly
                # val is a list [x_regions, y_region, y_value, degree]
                region = val[0]
                y_region = region[-2]
                rule_base.append(region[:-2])
            else:
                # create a new y region using triangular membership function
                # compute y_c (center of new y region)
                # w = weight of the rules = degree
                w = 0
                y_c = 0
                # now i is a list [x_regions, y_region, y_value, degree]
                for i in val:
                    y_c += i[-2] * i[-1]
                    w += i[-1]
                # y_c is the weighted average of y_value of each rules in the group
                y_c = y_c / w
                
                # compute sigma, the variance of y_c
                sigma = 0
                for i in val:
                    sigma += np.abs(i[-2] - y_c) * i[-1]
                sigma = sigma / w
                # for small sigma, we have issues using scikit-fuzzy to build the membership function
                # we thus remedy it here by assigning a large enough value temporarily
                if sigma < 0.1:
                    sigma = 0.1
                
                # build fuzzy region around y_c using triangle membership function (can be 
                # further customized)
                ant_con = self.list_obj
                r = "region" + str(key)
                ant_con[-1][r] = fuzz.trimf(ant_con[-1][r].universe, [y_c - sigma, y_c, y_c + sigma])

                # add rule into rule_base
                new_rule = val[0][:-3] + [r]
                rule_base.append(new_rule)
        print("{} final rules are built.".format(len(rule_base)))
        
        self.rule_base = rule_base


    def semiavg_select(self, rule_base):
        """
        If conflict, select a y region by averaging over y.
        Only supported if y region names are consecutive integers.
        e.g. ['1', '2', '3', ...]

        Parameter
        ---------
        rule_base: list
            A empty list for final rule base.
        """
        groups = self.group

        # loop over each group
        for key, val in groups.items():
                
            # check if only one rule in the group
            if len(val) == 1:
                # add it to rule_base directly
                # val is a list [x_regions, y_region, y_value, degree]
                region = val[0]
                y_region = region[-2]
                rule_base.append(region[:-2])
            else:
                # weighted average over y
                y_s = 0
                w = 0
                for v in val:
                    y_s += int(v[-3]) * v[-1]
                    w += v[-1]
                y_avg = y_s / w
                
                # add rule into rule_base
                new_rule = val[0][:-3] + [str(int(np.round(y_avg)))]
                rule_base.append(new_rule)
        print("{} final rules are built.".format(len(rule_base)))
        
        self.rule_base = rule_base


