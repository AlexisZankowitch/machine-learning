from sklearn import cross_validation


def mean_calculation(scores, instances):
    mean_table = []
    for i in range(0, len(instances)):
        m = 0
        for j in range(0, len(scores)):
            m += scores[j][i]
        m /= len(scores)
        mean_table.append(m)
    return mean_table


def scores_calculation(instances, my_tree, target, nb_cv):
    # TODO dynamically this table
    # , [], [], [], [], [], [], [], [], [], []
    scores = [[], [], [], [], [], [], [], [], [], []]
    for y in range(0, len(scores)):
        for x in range(0, len(instances)):
            # run a 5 fold cross validation on this model using the full census data
            scores_x_entropy = cross_validation.cross_val_score(my_tree, instances[x], target[x],
                                                                cv=nb_cv)                                                             
                                                                
            scores[y].append(scores_x_entropy.mean())
            # Show entropy accuracy
    return scores

