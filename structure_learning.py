from pgmpy.estimators import HillClimbSearch, ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork
from config import VARIABLES, WHITELIST, BLACKLIST

def learn_structure(data):
    data_for_structure = data[VARIABLES].copy()
    
    expert = ExpertKnowledge(required_edges=WHITELIST, forbidden_edges=BLACKLIST)
    
    hc = HillClimbSearch(data_for_structure)
    best_model = hc.estimate(scoring_method='bdeu', expert_knowledge=expert, max_indegree=4)
    return DiscreteBayesianNetwork(best_model.edges())