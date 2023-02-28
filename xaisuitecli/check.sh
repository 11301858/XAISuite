#!/bin/sh

source import.sh

if [[ $1 == "--model" ]]
then
    list="SVC NuSVC LinearSVC SVR NuSVR LinearSVR AdaBoostClassifier AdaBoostRegressor BaggingClassifier BaggingRegressor ExtraTreesClassifier ExtraTreesRegressor GradientBoostingClassifier GradientBoostingRegressor RandomForestClassifier RandomForestRegressor StackingClassifier StackingRegressor VotingClassifier VotingRegressor HistGradientBoostingClassifier HistGradientBoostingRegressor GaussianProcessClassifier GaussianProcessRegressor IsotonicRegression KernelRidge LogisticRegression LogisticRegressionCV PassiveAgressiveClassifier Perceptron RidgeClassifier RidgeClassifierCV SGDClassifier SGDOneClassSVM LinearRegression Ridge RidgeCV SGDRegressor ElasticNet ElasticNetCV Lars LarsCV Lasso LassoCV LassoLars LassoLarsCV LassoLarsIC OrthogonalMatchingPursuit OrthogonalMatchingPursuitCV ARDRegression BayesianRidge MultiTaskElasticNet MultiTaskElasticNetCV MultiTaskLasso MultiTaskLassoCV HuberRegressor QuantileRegressor RANSACRegressor TheilSenRegressor PoissonRegressor TweedieRegressor GammaRegressor PassiveAggressiveRegressor BayesianGaussianMixture GaussianMixture OneVsOneClassifier OneVsRestClassifier OutputCodeClassifier ClassifierChain RegressorChain MultiOutputRegressor MultiOutputClassifier BernoulliNB CategoricalNB ComplementNB GaussianNB MultinomialNB KNeighborsClassifier KNeighborsRegressor BernoulliRBM MLPClassifier MLPRegressor DecisionTreeClassifier DecisionTreeRegressor ExtraTreeClassifier ExtraTreeRegressor"
    if echo $list | grep -w -q $2
    then 
        echo Model is valid
        exit 0
    else
        echo Model is not valid
        exit 1
    fi

elif [[ $1 == "--data" ]]
then
    if [ "${2: -4}" == ".csv" -a -e $2 ]
    then
      echo Data is valid
      exit 0
    else
        list="diabetes iris digits californiahousing"
        if echo $list | grep -w -q $2
        then 
            echo Data is valid
            exit 0
        else
            echo Data is not valid
            exit 1
        fi
    fi
elif [[ $1 == "--explainers" ]]
then
    list="shap lime pdp ale shap sensitivity mace"
    if echo $list | grep -w -q $2
    then 
        echo Explainer is valid
        exit 0
    else
        echo Explainer is not valid
        exit 1
    fi
fi