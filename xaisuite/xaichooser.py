def train_and_explainModel(model:str, tabular_data:Tabular, x_ai:list, indexList:list = [], scale:bool = True, verbose:bool = False): # Returns the model function
    '''A function that attempts to train and explain a particular sklearn model.
    Parameters:
    model:str | Name of Model
    tabular_data:Tabular | Tabular object representing data set to be used in training
    x_ai:list | List of explanatory models to be used
    indexList:list = [] | Specific test data instance to be explained, by default empty (indicating all instances should be explained)
    scale:bool = True | Whether data should be scaled before training
    verbose:bool = False | Whether debugging information should be printed
    
    Returns:
    The learning model
    '''
   
    try:
      modeler= eval(model + "()") #Create model function from provided model name. This will not work if model is not part of sklearn library or is unsupervised.
    except:
      print("Provided model name is incorrect or is not part of sklearn library. Only supervised learning models in sklearn are supported. Refer to models by their associated functions. For example, if you want to use support vector regression, pass in \"SVR\"")
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
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)

   
    modeler.fit(x_train, y_train) #Train model
    
    if(verbose): #Useful debugging data

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
      return modeler #Return the trained model for future use by user
        
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
    )
    
    # Generate explanations
    test_instances = test_data #[0:5] For first 5 instances, disabled
    local_explanations = explainers.explain(X=test_instances)
    global_explanations = explainers.explain_global(
    params={"pdp": {"features": (tabular_data.to_pd().drop([tabular_data.target_column], axis=1)).columns}}
    )
    print("GENERATING EXPLANATIONS FOR MODEL...\n")
    if (verbose):
      for k,v in local_explanations.items(): #For debugging, print dictionary containing local explanations
        print(k, v.get_explanations()) 

    
    try:
      for i in range (len(x_ai)): #For each explanatory method requested by user
          print(x_ai[i].capitalize() + " Results:") #Print name of explanatory method as title
          if (x_ai[i] in local_explanations.keys()): #If the explanatory method is local (it explains one instance at a time)
            if (verbose):
                print(local_explanations[x_ai[i]].get_explanations()) #Print explanations if debugging

            #Store explanations in CSV file
            keys = local_explanations[x_ai[i]].get_explanations()[0].keys() 

            CSVFile = x_ai[i] + " ImportanceScores - " + model + " " + tabular_data.target_column + ".csv"

            with open(CSVFile, 'a', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(local_explanations[x_ai[i]].get_explanations())

            #Now, show explanation graphs for requested instances
            if (not indexList): #If no requested instances provided
              indexList = range(0,len(test_instances)) #Show all instances
            for index in indexList:
              local_explanations[x_ai[i]].ipython_plot(index) #Otherwise, show requested instances
          else: #Explanatory method is global (for the entire dataset)
              try:
                  global_explanations[x_ai[i]].ipython_plot() #Show explanation graph
              except:
                  raise ValueError(x_ai[i] + " is not a valid explanatory method or was not requested.") #If we get to this line, explanation method is not valid because it is not present in either local explanations or global explanations dictionaries
    except Exception as e:
      print("EXPLANATIONS FAILED - " + str(e)) #Something went wrong

    return modeler #In the end, return trained model for future use by user

