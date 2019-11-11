import os
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

if __name__ == "__main__":
    # Load boosted tree
    bst = pickle.load(open(os.path.join(BASE_DIR, 'saved_models/20191110194302_gbtree.model'), 'rb'))

    # Plot feature importance
    xgb.plot_importance(bst.get_booster())
    plt.show()

    # Plot tree
    xgb.plot_tree(bst.get_booster())
    plt.show()
