from .imports import*

def train_and_explainModel(model:str, tabular_data:Tabular, x_ai:list, indexList:list = [], scale:bool = True, scaleType:str = "StandardScaler", addendum:str = "", verbose:bool = False, **modelSpecificArgs): # Returns the model function and scaler (if applicable)
    ''' A function that attempts to train and explain a particular sklearn model.
    
    :param str model: Name of Model
    :param Tabular tabular_data: Tabular object representing data set to be used in training
    :param list x_ai: List of explanatory models to be used
    :param list indexList: Specific test data instance to be explained, by default empty (indicating all instances should be explained)
    :param bool scale: Whether data should be scaled before training, by default True
    :param str scaleType: Default Scaler type is "StandardScaler". Example: Use "MinMaxScaler" for MultinomialNB model.
    :param str addendum: Added string to explanation files in case multiple models are being trained and explained within the same directory, to prevent overwriting. By default, empty string.
    :param bool verbose: Whether debugging information should be printed, by default False
    :param ``**modelSpecificArgs``: Specific arguments to pass on to the model function
    
    :return: The learning model and The scaler (if applicable) if user wants to predict more values. In List format
    :rtype: list  
    '''
   
    returnList = []
    
    acceptedModels = {"SVC": "sklearn.svm.SVC", "NuSVC": "sklearn.svm.NuSVC", "LinearSVC": "sklearn.svm.LinearSVC", "SVR": "sklearn.svm.SVR", "NuSVR": "sklearn.svm.SVR", "LinearSVR": "sklearn.svm.LinearSVR", 
                 "AdaBoostClassifier": "sklearn.ensemble.AdaBoostClassifier", "AdaBoostRegressor": "sklearn.ensemble.AdaBoostRegressor", "BaggingClassifier": "sklearn.ensemble.BaggingClassifier", "BaggingRegressor": "sklearn.ensemble.BaggingRegressor",
                 "ExtraTreesClassifier": "sklearn.ensemble.ExtraTreesClassifier", "ExtraTreesRegressor": "sklearn.ensemble.ExtraTreesRegressor", 
                 "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier", "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
                 "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier", "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
                 "StackingClassifier": "sklearn.ensemble.StackingClassifier", "StackingRegressor": "sklearn.ensemble.StackingRegressor",
                 "VotingClassifier": "sklearn.ensemble.VotingClassifier", "VotingRegressor": "sklearn.ensemble.VotingRegressor",
                 "HistGradientBoostingClassifier": "sklearn.ensemble.HistGradientBoostingClassifier", "HistGradientBoostingRegressor": "sklearn.ensemble.HistGradientBoostingRegressor",
                 "GaussianProcessClassifier": "sklearn.gaussian_process.GaussianProcessClassifier", "GaussianProcessRegressor": "sklearn.gaussian_process.GaussianProcessRegressor", 
                 "IsotonicRegression": "sklearn.isotonic.IsotonicRegression", "KernelRidge": "sklearn.kernel_ridge.KernelRidge", 
                 "LogisticRegression": "sklearn.linear_model.LogisticRegression", "LogisticRegressionCV": "sklearn.linear_model.LogisticRegressionCV",
                 "PassiveAgressiveClassifier": "sklearn.linear_model.PassiveAggressiveClassifier", "Perceptron": "sklearn.linear_model.Perceptron", 
                  "RidgeClassifier": "sklearn.linear_model.RidgeClassifier", "RidgeClassifierCV": "sklearn.linear_model.RidgeClassifierCV",
                  "SGDClassifier": "sklearn.linear_model.SGDClassifier", "SGDOneClassSVM": "sklearn.linear_model.SGDOneClassSVM", 
                  "LinearRegression": "sklearn.linear_model.LinearRegression", "Ridge": "sklearn.linear_model.Ridge", 
                  "RidgeCV": "sklearn.linear_model.RidgeCV", "SGDRegressor": "sklearn.linear_model.SGDRegressor",
                  "ElasticNet": "sklearn.linear_model.ElasticNet", "ElasticNetCV": "sklearn.linear_model.ElasticNetCV",
                  "Lars": "sklearn.linear_model.Lars", "LarsCV": "sklearn.linear_model.LarsCV", 
                  "Lasso": "sklearn.linear_model.Lasso", "LassoCV": "sklearn.linear_model.LassoCV",
                  "LassoLars": "sklearn.linear_model.LassoLars", "LassoLarsCV": "sklearn.linear_model.LassoLarsCV",
                  "LassoLarsIC": "sklearn.linear_model.LassoLarsIC", "OrthogonalMatchingPursuit": "sklearn.linear_model.OrthogonalMatchingPursuit",
                  "OrthogonalMatchingPursuitCV": "sklearn.linear_model.OrthogonalMatchingPursuitCV", "ARDRegression": "sklearn.linear_model.ARDRegression",
                  "BayesianRidge": "sklearn.linear_model.BayesianRidge", "MultiTaskElasticNet": "sklearn.linear_model.MultiTaskElasticNet", 
                  "MultiTaskElasticNetCV": "sklearn.linear_model.MultiTaskElasticNetCV", "MultiTaskLasso": "sklearn.linear_model.MultiTaskLasso",
                  "MultiTaskLassoCV": "sklearn.linear_model.MultiTaskLassoCV", "HuberRegressor": "sklearn.linear_model.HuberRegressor",
                  "QuantileRegressor": "sklearn.linear_model.QuantileRegressor", "RANSACRegressor": "sklearn.linear_model.RANSACRegressor",
                  "TheilSenRegressor": "sklearn.linear_model.TheilSenRegressor", "PoissonRegressor": "sklearn.linear_model.PoissonRegressor",
                  "TweedieRegressor": "sklearn.linear_model.TweedieRegressor", "GammaRegressor": "sklearn.linear_model.GammaRegressor", 
                  "PassiveAggressiveRegressor": "sklearn.linear_model.PassiveAggressiveRegressor", "BayesianGaussianMixture": "sklearn.mixture.BayesianGaussianMixture",
                  "GaussianMixture": "sklearn.mixture.GaussianMixture", 
                  "OneVsOneClassifier": "sklearn.multiclass.OneVsOneClassifier", "OneVsRestClassifier": "sklearn.multiclass.OneVsRestClassifier", 
                  "OutputCodeClassifier": "sklearn.multiclass.OutputCodeClassifier", "ClassifierChain": "sklearn.multioutput.ClassifierChain", 
                   "RegressorChain": "sklearn.multioutput.RegressorChain",  "MultiOutputRegressor": "sklearn.multioutput.MultiOutputRegressor",
                   "MultiOutputClassifier": "sklearn.multioutput.MultiOutputClassifier", "BernoulliNB": "sklearn.naive_bayes.BernoulliNB", 
                  "CategoricalNB": "sklearn.naive_bayes.CategoricalNB", "ComplementNB": "sklearn.naive_bayes.ComplementNB", 
                  "GaussianNB": "sklearn.naive_bayes.GaussianNB", "MultinomialNB": "sklearn.naive_bayes.MultinomialNB", 
                  "KNeighborsClassifier": "sklearn.neighbors.KNeighborsClassifier", "KNeighborsRegressor": "sklearn.neighbors.KNeighborsRegressor", 
                  "BernoulliRBM": "sklearn.neural_network.BernoulliRBM", "MLPClassifier": "sklearn.neural_network.MLPClassifier", "MLPRegressor": "sklearn.neural_network.MLPRegressor",  
                  "DecisionTreeClassifier": "sklearn.tree.DecisionTreeClassifier", "DecisionTreeRegressor": "sklearn.tree.DecisionTreeRegressor",
                  "ExtraTreeClassifier": "sklearn.tree.ExtraTreeClassifier", "ExtraTreeRegressor": "sklearn.tree.ExtraTreeRegressor"
                 }
    
    try:
      assert model in acceptedModels.keys()
      import eval(acceptedModels.get(model))
      modeler= eval(model + "( **modelSpecificArgs )") #Create model function from provided model name. This will not work if model is not part of sklearn library or is unsupervised.
    except Exception as e:
      print("Provided model name is incorrect or is not part of sklearn library. Only supervised learning models in sklearn are supported. Refer to models by their associated functions. For example, if you want to use support vector regression, pass in \"SVR\". \n Error message: " + str(e))
      log = open("Failed_Models.txt", 'a', newline = '\n')
      log.write(model + ": " + str(e) + "\n") 
      return None
    

    #Tranform data
    transformer = TabularTransform(
            target_transform=Identity()
        ).fit(tabular_data)
    x = transformer.transform(tabular_data)


    #Create training and testing
    x_train, x_test, y_train, y_test = \
        train_test_split(x[:, :-1], x[:, -1], test_size = 0.2, random_state = 0)

    #Scale data
    if (scale):
        scaler = eval(scaleType + "()")

        x_train = scaler.fit_transform(x_train)
        
        returnList.append(scaler)

        x_test = scaler.transform(x_test)

   
    try:
        modeler.fit(x_train, y_train) #Train model
    except Exception as e:
        print("Provided model could not be fit to data. Error message: " + str(e))
        log = open("Failed_Models.txt", 'a', newline = '\n')
        log.write(model + ": " + str(e) + "\n") 
        return None
    
    returnList.append(modeler)
    
    if(verbose): #Useful debugging data
        
        print(returnList)
        comparison_data = pd.DataFrame(data = list(zip(modeler.predict(x_test), y_test, [a_i - b_i for a_i, b_i in zip(modeler.predict(x_test), y_test)])), columns = [model + ' Predicted ' + tabular_data.target_column, 'Actual ' + tabular_data.target_column, "Difference"]) 
        print("LEARNING FROM DATA...\n ") #Redundant title to separate output from other possible debugging messages
        print(comparison_data) #Print results of model training with Actual Target and Predicted Target
        filepath = Path('modelresults.csv')  
        comparison_data.to_csv(filepath) #Save data to CSV file

        try:
            score = modeler.score(x_test, y_test) #Calculates R^2 value by comparing predicted target values and actual target values
            print(model + " score is " + str(score)) #Print model score
        except Exception as e:
            print("Could not retrieve model score. " + str(e)) #This will occur if an error occurred while calculating score or if model does not support score calculation

    if (len(x_ai) == 0): #List of explanatory methods is blank
      print("NO EXPLANATIONS REQUESTED BY USER") #Print that no explanations were requested by user
      return returnList #Return the trained model for future use by user
        
    # Convert the transformed data back to Tabular instances
    train_data = transformer.invert(x_train)
    test_data = transformer.invert(x_test)
    
    
    #Again, works for only sklearn models. Checks whether model is a regressor or classifier. No unsupervised learning models allowed. 
    if (is_regressor(eval(model + "()"))): 
        detectedmode = 'regression'
    elif(is_classifier(eval(model + "()"))):
        detectedmode = 'classification'
    else:
        raise ValueError('Model provided is not supervised.')
    
    #Create Tabular Explainer object in preparation for generating explanations
    explainers = TabularExplainer(
        explainers=x_ai,
        mode=detectedmode,
        data=train_data,
        model=modeler,
        preprocess= lambda z: transformer.transform(z),
        params = {"shap": {"link": "identity"}}
    )
    
    # Generate explanations
    test_instances = test_data #[0:5] For first 5 instances, disabled
    local_explanations = explainers.explain(X=test_instances)
    global_explanations = explainers.explain_global(params = {"pdp": {"features": (tabular_data.to_pd().drop([tabular_data.target_column], axis=1)).columns}})
    if (verbose):
      print("GENERATING EXPLANATIONS FOR MODEL...\n")
      for k,v in local_explanations.items(): #For debugging, print dictionary containing local explanations
        print(k, v.get_explanations()) 

    
    try:
      for i in range (len(x_ai)): #For each explanatory method requested by user
          if(verbose):
            print(x_ai[i].upper() + " Results:") #Print name of explanatory method as title
          if (x_ai[i] in local_explanations.keys()): #If the explanatory method is local (it explains one instance at a time)
            if (verbose):
                print(local_explanations[x_ai[i]].get_explanations()) #Print explanations if debugging

            #Store explanations in CSV file
            keys = local_explanations[x_ai[i]].get_explanations()[0].keys() 

            CSVFile = x_ai[i] + " ImportanceScores - " + model + " " + tabular_data.target_column + addendum + ".csv"

            with open(CSVFile, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(local_explanations[x_ai[i]].get_explanations())

            #Now, show explanation graphs for requested instances (only verbose = True)
            if (verbose):
                if (not indexList): #If no requested instances provided
                  indexList = range(0,len(test_instances)) #Show all instances
                for index in indexList:
                  local_explanations[x_ai[i]].ipython_plot(index) #Otherwise, show requested instances
          else: #Explanatory method is global (for the entire dataset)
              try:
                 if(verbose):
                  global_explanations[x_ai[i]].ipython_plot() #Show explanation graph
              except:
                  raise ValueError(x_ai[i] + " is not a valid explanatory method or was not requested.") #If we get to this line, explanation method is not valid because it is not present in either local explanations or global explanations dictionaries
    except Exception as e:
      print("EXPLANATIONS FAILED - " + str(e)) #Something went wrong

    return returnList #In the end, return trained model and scaler for future use by user

