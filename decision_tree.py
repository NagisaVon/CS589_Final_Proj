import math, random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def entropy(data):
    '''
    list of classes -> entropy
    '''
    cnt = list(Counter(data).values())
    total = len(data)
    return sum([-k/total * math.log(k/total, 2) for k in cnt])


def gini(data):
    cnt = list(Counter(data).values())
    total = len(data)
    return 1 - sum([k/total * (k/total) for k in cnt])



def entropy_in_list(data, attr_list, attr_type, attr_opt, tag_col, attr_list_limited, algo):
    '''
    house_data(w/ tags)->entropy_of_all_attributes[]
    '''
    # attr_count is not calcutated from the attr_list, which removed "class" attribute
    # but from the data demension, which includes the "class" attribute
    # this is to make sure the argmin/argmax function works
    attr_count = len(data.T)
    total_entry = len(data)
    ent_list = []
    numerial_split_point = []
    # for each attributes, calculate information gain
    for i in range(attr_count):
        # to use argmin later, keep all attribute in the list
        # skip if not in attr_list
        # (this is no longer neseccary because we don't delete attributes from data
        # but i keep it here just in case)
        # also skip if only look at m attributes, but i is not is attr_list_limited
        isExcluded = (attr_list_limited is not None) and (not i in attr_list_limited)
        if(not i in attr_list or attr_type[i] == 'class' or isExcluded):
            ent_list.append(1000)  # infinite high entropy
            numerial_split_point.append(None)
            continue
        elif (attr_type[i] == 'numerical'):
            best_ent = 1000
            best_split_point = -1
            for split_ind in range(len(attr_opt[i])):
                # 0 for <=, 1 for >
                option_tags = {0: [], 1: []}
                for row in data:
                    if row[i] <= attr_opt[i][split_ind]:
                        option_tags[0].append(row[tag_col])
                    else:
                        option_tags[1].append(row[tag_col])
                ent = sum(len(option_tags[opt])/total_entry * algo(option_tags[opt]) for opt in option_tags)
                if (ent < best_ent):
                    best_ent = ent
                    best_split_point = attr_opt[i][split_ind]
            ent_list.append(best_ent)  
            numerial_split_point.append(best_split_point)
        elif (attr_type[i] == 'categorical'): 
            # init dictionary with keys
            # don't use the from the dict.fromkeys function
            # that will link all keys to the same array
            option_tags = {attr_opt[i][j]: [] for j in range(len(attr_opt[i]))}
            # a optimization, make sure only go through the data once
            for row in data:
                option_tags[row[i]].append(row[tag_col])
            ent = sum(len(option_tags[opt])/total_entry * algo(option_tags[opt]) for opt in option_tags)
            ent_list.append(ent)
            numerial_split_point.append(None)
    return ent_list, numerial_split_point


class Node:
    # tag means different classes, since class is a reserved keyword
    def __init__(self, depth, isLeaf=False, attr_index=None, num_split=None, tag=None, info_gain=None):
        self.depth = depth
        self.attr_index = attr_index
        self.num_split = num_split
        self.isLeaf = isLeaf
        self.tag = tag
        self.children = []
        self.option = None
        self.parent = None
        self.info_gain = info_gain
        if (num_split is None):
            self.type = "categorical"
        else:
            self.type = "numerical"

    def print_tree(self, attribute_list, indent=0):
        ind_str = ' ' * indent
        if self.isLeaf:
            print( ind_str + "Leaf:CLASS" , self.tag, "OPTION", self.option , "DEPTH" , self.depth)
        else:
            print( ind_str , attribute_list[self.attr_index] ,"CLASS", self.tag,  "NSPLIT", self.num_split, "OPTION" , self.option , "GAIN",self.info_gain, "DEPTH" , self.depth)
            for child in self.children:
                child.print_tree(attribute_list, indent+2)


def build_decision_tree(data, attr_list, attr_type, attr_opt, tag_col, algo, minimal_size_for_split, minimal_gain, maximal_depth, only_m_attr, _depth=0):

    # If there are no more data
    if (not data.any()):
        return Node( _depth, isLeaf=True)

    tags = data.T[tag_col]
    # keep track of a majority tag in case the leaf is None
    # .most_common()[0][0] returns the most common tag (key of a dict)
    most_common_tag = Counter(tags).most_common()[0][0]
    original_entropy = entropy(tags)

    # If there are no more attributes that can be tested, 
    # or depth larger than maximal_depth
    # or only one class left
    # return the most common tag
    if (len(attr_list) == 0 
        or _depth>=maximal_depth 
        or len(data) <= minimal_size_for_split 
        or original_entropy == 0): 
        return Node( _depth, isLeaf=True, tag=most_common_tag)

    # for random forest, only consider m attributes
    # always pass only_m_attr to avoid calculate sqrt multiple times
    # make sure attr_list has enough length (it will always have since no more attributes removed)
    # and "class" is already removed from attr_list
    if only_m_attr != 0 and only_m_attr < len(attr_list): 
        attr_list_limited = random.sample(attr_list, only_m_attr)
    else:
        attr_list_limited = None

    # get the index of the attribute with highest information gain
    if algo=='gini':
        gini_val, num_split = entropy_in_list(data, attr_list, attr_type, attr_opt, tag_col, attr_list_limited, gini)
        decided_attr = np.argmin(gini_val)
        m_info_gain = np.min(gini_val)
        # information gain too low
        if (m_info_gain)  <= minimal_gain:
            return Node( _depth, isLeaf=True, tag=most_common_tag)
        decided_num_split = num_split[decided_attr]
    elif algo=='entropy':
        entropy_list, num_split = entropy_in_list(data, attr_list, attr_type, attr_opt, tag_col, attr_list_limited, entropy)
        # this subtraction is not necessary, just to be justify the name 'info_gain'
        info_gain = [original_entropy - ent for ent in entropy_list]
        m_info_gain = np.max(info_gain);
        # information gain too low
        if (m_info_gain) <= minimal_gain:
            return Node( _depth, isLeaf=True, tag=most_common_tag)
        decided_attr = np.argmax(info_gain)
        decided_num_split = num_split[decided_attr]
    
    nd = Node( _depth, isLeaf=False, attr_index=decided_attr, 
            num_split=decided_num_split, 
            tag=most_common_tag, 
            info_gain=m_info_gain)
    
    # Ternary operation in python
    isNumerical = True if (decided_num_split is not None) else False
    # build subtree for each possible option
    # Numerical 
    if isNumerical:
        options = [0, 1] # the attr_opt for numerical attribute is storing possible splits instead
    else: # Catetorical 
        options = attr_opt[decided_attr]
    
    for opt in options:
        # filtered data
        if isNumerical:
            if opt == 0:
                # a filter, if at the decided_attr, the value is <= decided_num_split, then that row is kept
                filtered_data = data[data.T[decided_attr] <= decided_num_split]
            else: 
                filtered_data = data[data.T[decided_attr] > decided_num_split]
        else:
            filtered_data = data[data.T[decided_attr] == opt]
        subtree = build_decision_tree(filtered_data, attr_list, attr_type, attr_opt, tag_col, algo, minimal_size_for_split, minimal_gain, maximal_depth, only_m_attr, _depth+1) 
        subtree.option = opt
        nd.children.append(subtree)
        subtree.parent = nd
    return nd


def find_matched_child(row, node):
    # row[node.attr_index] is the the data of the attribute of the node
    if node.type == "numerical":
        if row[node.attr_index] <= node.num_split:
            return node.children[0]
        else: 
            return node.children[1]
    else:  # categorical
        for child in node.children:
            if row[node.attr_index] == child.option:
                return child
    return None;


def predict(row, node):
    child = find_matched_child(row, node)
    if child is None:
        return None;
    if child.isLeaf:
        if child.tag == None:
            return node.tag
        else:
            return child.tag
    else:
        return predict(row, child)


def evaluate_acc(data, tree, tag_col):
    correct = 0
    for i in range(len(data)):
        result = predict(data[i], tree)
        if data[i][tag_col] == result:
            correct += 1
    return correct / len(data)


def dispatch_decision_tree(data, attr, attr_type, attr_opt, tag_col, algo, minimal_size_for_split=0., minimal_gain=0., maximal_depth=10000,random_state=42, printTree=False, only_m_attr:int=0):
    train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=random_state)
    # build a attr_list not including the class attribute
    attr_list = [i for i in range(len(attr)) if attr_type[i] != "class"]
    tree = build_decision_tree(train, attr_list, attr_type, attr_opt, tag_col, algo, minimal_size_for_split, minimal_gain, maximal_depth, only_m_attr)
    if(printTree):
        tree.print_tree(attr)
    return evaluate_acc(train, tree, tag_col), evaluate_acc(test, tree, tag_col)


