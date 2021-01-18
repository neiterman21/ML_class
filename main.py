# using python3.6.10
# need to install numpy and pandas

# HC Data H_k:
"""
(1, 'p1_distance') Empirical error: 0.5501538461538462 True error: 0.5495384615384618
(1, 'p2_distance') Empirical error: 0.5480000000000002 True error: 0.5464615384615389
(1, 'p_inf_distance') Empirical error: 0.5486153846153847 True error: 0.5464615384615388
(3, 'p1_distance') Empirical error: 0.5507692307692311 True error: 0.5650769230769233
(3, 'p2_distance') Empirical error: 0.5527692307692312 True error: 0.5650769230769231
(3, 'p_inf_distance') Empirical error: 0.5504615384615391 True error: 0.563846153846154
(5, 'p1_distance') Empirical error: 0.5506153846153854 True error: 0.5544615384615387
(5, 'p2_distance') Empirical error: 0.5501538461538464 True error: 0.5579999999999999
(5, 'p_inf_distance') Empirical error: 0.5490769230769236 True error: 0.5598461538461542
(7, 'p1_distance') Empirical error: 0.5421538461538465 True error: 0.5526153846153848
(7, 'p2_distance') Empirical error: 0.5447692307692311 True error: 0.5490769230769235
(7, 'p_inf_distance') Empirical error: 0.5483076923076928 True error: 0.5460000000000003
(9, 'p1_distance') Empirical error: 0.5469230769230773 True error: 0.5490769230769237
(9, 'p2_distance') Empirical error: 0.5460000000000005 True error: 0.5449230769230771
(9, 'p_inf_distance') Empirical error: 0.5441538461538468 True error: 0.5438461538461542
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
            error_df = result_df == test_labels
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
