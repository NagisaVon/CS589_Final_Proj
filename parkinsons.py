# Evaluate The Oxford Parkinson’s Disease Detection Dataset
# 22 numerical attributes, binary output,
# 195 instances 
# Algorithm: KNN/Random Forest/NN

from load_data import * 

# load data and the metadata
parkinsons_data, parkinsons_attr = load_data("datasets/parkinsons.csv", csv_delimiter=',')
parkinsons_attr_type = ["numerical","numerical","numerical","numerical","numerical",
        "numerical","numerical","numerical","numerical","numerical",
        "numerical","numerical","numerical","numerical","numerical",
        "numerical","numerical","numerical","numerical","numerical","numerical","numerical","class"]
parkinsons_attr_dict = build_attribute_dict(parkinsons_attr, parkinsons_attr_type)
parkinsons_attr_options = get_possible_options(parkinsons_data, parkinsons_attr_type)
parkinsons_class_col = -1


