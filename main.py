# using python3.6.10
# need to install numpy and pandas

# HC Data H_k:
"""
(1, 'p1_distance') Empirical error: 0.9713846153846143 True error: 0.5543076923076925
(1, 'p2_distance') Empirical error: 0.9712307692307681 True error: 0.5515384615384619
(1, 'p_inf_distance') Empirical error: 0.9713846153846143 True error: 0.5513846153846156
(3, 'p1_distance') Empirical error: 0.7772307692307691 True error: 0.571538461538462
(3, 'p2_distance') Empirical error: 0.7710769230769227 True error: 0.5693846153846157
(3, 'p_inf_distance') Empirical error: 0.7666153846153844 True error: 0.5643076923076926
(5, 'p1_distance') Empirical error: 0.7221538461538458 True error: 0.5530769230769235
(5, 'p2_distance') Empirical error: 0.7176923076923074 True error: 0.5530769230769235
(5, 'p_inf_distance') Empirical error: 0.7092307692307692 True error: 0.5543076923076928
(7, 'p1_distance') Empirical error: 0.6896923076923075 True error: 0.5480000000000004
(7, 'p2_distance') Empirical error: 0.6895384615384614 True error: 0.5484615384615386
(7, 'p_inf_distance') Empirical error: 0.6801538461538459 True error: 0.5480000000000004
(9, 'p1_distance') Empirical error: 0.6638461538461539 True error: 0.5509230769230773
(9, 'p2_distance') Empirical error: 0.6633846153846154 True error: 0.5492307692307695
(9, 'p_inf_distance') Empirical error: 0.6598461538461539 True error: 0.5449230769230771
"""


##############################
from KNN import *
import readData
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import itertools

def calc_error(rules, data):
    err = 0
    for i in range(1, len(data.index)):
        w_vote = 0
        for rule in rules:
            w_vote += rule[1] * rule[0].predict([data["c1"][i], data["c2"][i]])
        if np.sign(w_vote) != data["label"][i]:
            err += 1
    return err


def run_single_experiment(data,labels,seed):
    """
    Perform a single iteration of 15 runs.
    """
    idx =  np.random.RandomState(seed=seed).permutation(data.index)
    data= data.reindex(idx)
    labels =labels.reindex(idx)
    (train_data, test_data) = np.vsplit(data, 2)
    train_labels = labels.head(65)
    test_labels  = labels.tail(65)
    test_data = test_data.reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    test_labels = test_labels.reset_index(drop=True)
    Ks = [1,3,5,7,9]
    Ps = [p1_distance,p2_distance,p_inf_distance]
    emp_error = []
    true_error = []
    for k in Ks:
        for p in Ps:
            #true error
            result_df = test_data.apply(lambda row: classify(row, train_data, train_labels,p, k), axis=1)
            error_df = result_df == test_labels
          #  print( "true: k = " + str(k) , "p= " + p.__name__ , "acc= ", error_df.value_counts(normalize=True).iloc[0])
            true_error.append(error_df.value_counts(normalize=True).iloc[0])
            #empirical error
            result_df = train_data.apply(lambda row: classify(row, train_data, train_labels,p, k), axis=1)
            error_df = result_df == train_labels
            emp_error.append(error_df.value_counts(normalize=True).iloc[0])
           # print( "emp: k = " + str(k) , "p= " + p.__name__ , "acc= ", error_df.value_counts(normalize=True).iloc[0])
   
    return np.asarray(emp_error), np.asarray(true_error)


def run_experiment(data,labels, iterations=100):
    """
    Run KNN on data where every iteration 15 runs are performed.
    k=[1,3,5,7,9] p=[1,2,inf] runs=kXp
    iterations default value is 100.
    Note that this function runs using multiple cores, adjust below according to your specs
    """
    emp_error = np.zeros(15)
    true_error = np.zeros(15)
    num_cores = multiprocessing.cpu_count()
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(run_single_experiment)(data,labels,i) for i in range(iterations)
    )
    for res in processed_list:
        
        emp_error = np.add(emp_error, res[0])
        true_error = np.add(true_error, res[1])
         
    emp_error /= iterations
    true_error /= iterations
    return emp_error, true_error


def main():
    hbt , labels = readData.hbt_data("HC_Body_Temperature.txt")
    emp_err, true_err = run_experiment(hbt,labels, 100)
    Ks = [1,3,5,7,9]
    Ps = ["p1_distance","p2_distance","p_inf_distance"]
    
    for emp, tru , desc in zip(emp_err, true_err , itertools.product(Ks,Ps)):
        print(desc,
            "Empirical error:",
            emp,
            "True error:",
            tru,
        )


if __name__ == "__main__":
    main()
