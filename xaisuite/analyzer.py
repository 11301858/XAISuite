from .imports import*

def compare_explanations(filenames:list, showGraphics = True, verbose = False, **addendumkwargs): #Analyze the generated explanations in given files
  '''A function that analyzes and compares the explanations generated by train_and_explainModel.
    
    :param list filenames: File names with explanations (of the form "Explainer ImportanceScores - Model Target.csv")
    :param bool showGraphics = True: enables analysis graphics
    :params bool verbose = False: enables debugging dialogue
    :param ``**addendumkwargs``: Any additional columns to be added to analysis, such as dataset name. Any list to be added to graphics should be of the form addendumName = [addendumList]
    :return: None
    '''
  dataset = addendumkwargs["dataset"] if "dataset" in addendumkwargs else "" #If the user passes in the name of the dataset to be used as an addendum to the generated analysis CSV, store the value in a variable called dataset, eitherwise the adendum to the file name is empty
  if "dataset" in addendumkwargs: #If user passes in the name of the dataset, delete it from the addendumkwargs after sotring its value. All other addendumkwargs must be of the form addendumName = [addendumList]] and will be plotted on analysis graphics
    del addendumkwargs["dataset"]
  
  model = filenames[0].split()[3] #All files should have the same model, so we take the model name to be the third word in the file name of the first file. If you want to compare with other models, use the addendum option
  print(model.upper() + "\n" + "============") # Title
  print("There are " + str(len(filenames)) + " files.") #Clarify the user has passed in a certain number of files
  corrList = [] #Create a list to store the correlations between explainers if the number of explainers is 2
  try:
    df = pd.read_csv(filenames[0]) #Read the first file onto a dataframe
    df['features'][0] = ast.literal_eval(df['features'][0]) #Make sure to compare the string representation of the list of features for the first instance into an actual list
    for feature in sorted(df['features'][0]): #For each feature listed on the first instance (sorted so that this applies to the first instances for any model and explainer). We assume that all instances provide importance scores for all features
        if len(filenames) != 2: #If there are more than 2 explainers to compare (with 1 explainer, there will be nothing to compare)
          compare_explanationssinglef(filenames, feature, verbose, **addendumkwargs) #Compare the explanations for the specific features
        else: #If there are exactly 2 explainers to compare, then a single correlation can be found and stored
           corrList.append(compare_explanationssinglef(filenames, feature, verbose, **addendumkwargs)) #Compare the explanations for the specific features
    
    if len(filenames) == 2: #Find average of correlations if there are only 2 explainers to compare
      import statistics
      print("Average correlation is " + str(statistics.fmean(corrList)))
    
    if len(filenames) == 2: #Graphics and correlation information are only supported for 2 explainers
      print("Printing in-depth information since 2 explainers are provided.")
      try:
        data = pd.read_csv("featuresvsmodel" + dataset + ".csv") #Read the data storage file onto a dataframe if it already exists
      except Exception as e: #Data storage file not found or could not be retrieved
        with open("featuresvsmodel" + dataset + ".csv", 'w', newline='') as file: #Create the data storage file
          writer = csv.writer(file)
          writer.writerow(["Model"] + sorted(df['features'][0])) #Create the header for the file: Model | Feature 1 | Feature 2 etc. Note the order of the features is as they appear in the first file's first instance
        data = pd.read_csv("featuresvsmodel" + dataset + ".csv") #Now read the data storage file onto a dataframe
      finally: #Now that we have the data storage file created and read onto a dataframe
        data.loc[len(data.index)] = [model] + corrList # Add a new row with the model name and the explainer correlations for each feature
        data.set_index('Model', inplace=True, drop=True) #Set the index of the dataframe to the model name instead of 0, 1, 2, 3...
        display(data) #Display the dataframe
        plt.matshow(data) #Create a heatmap of correlations
        plt.title("Correlation between " + filenames[0].split()[0] + " and " + filenames[1].split()[0]) #For clarification, print the two explainers the user wants to compare
        #Set the axis labels for the heat map to be Features vs Models
        plt.xlabel("Features")
        plt.ylabel("Model")
        plt.xticks(ticks = range (0, len(df['features'][0])), labels = sorted(df['features'][0]))
        plt.yticks(ticks = range (0, len(data.index)), labels = data.index)
        plt.colorbar() #Create the color key for the heat map
        
        plt.show() if showGraphics == True else print("No graphics shown as requested.") #We construct the graph anyway but only show it if it is asked for
        data.to_csv("featuresvsmodel" + dataset + ".csv") #Write the specific model's explainer correlation scores for all features to the storage file in case the same dataset is trained on different models
        
      
  except Exception as e:
    print("An error occurred while analyzing the graph. " + str(e)) #Something went wrong
  

def compare_explanationssinglef(filenames:list, feature:str, verbose = False, **addendumkwargs): #Analyze the generated explanations in given files for a specific feature. If feature is not available for any instance or any filename, the importance is assumed to be 0, but behavior is unpredictable.
  '''A function that analyzes and compares the explanations generated by train_and_explainModel.
    
    :param list filenames: File names with explanations (of the form "Explainer ImportanceScores - Model Target.csv")
    :param str feature: Feature whose importance scores are to be compared
    :params bool verbose = False: enables debugging dialogue
    :param ``**addendumkwargs``: Any additional columns to be added to analysis. Each new parameter should be of the form addendumName = [addendumList]]
    :return: Correlation between different explainer files
    '''
  explainers = []
  features = [] #All files should have same features (though not necessarily in the same order) If a feature in one document is absent in the other, the feature's importance score is not considered, or considered to be zero, but behavior is unpredictable.
  model = filenames[0].split()[3] #All files should have the same model
  data = pd.DataFrame()
  for filename in filenames: #For exah explainer file
    try: 
        df = pd.read_csv(filename) #Load the explainer file onto a dataframe

        for i in range(len(df['features'])):
            if(verbose):
                print (df['features'][i])
            df['features'][i] = ast.literal_eval(df['features'][i]) #Make sure all the feature lists are in readable form
            #df.loc[:, ('features', i)] = ast.literal_eval(df['features'][i])

        for i in range(len(df['scores'])):
            if(verbose):
                print (df['features'][i])
            df['scores'][i] = ast.literal_eval(df['scores'][i]) #Make sure all the score lists are in readable form
            #df.loc[:, ('scores', i)] = ast.literal_eval(df['scores'][i])

        features = df['features']
        scores = df['scores']
        explainer = filename.split()[0]
        explainers.append(explainer)

        
        featurescore = []
        for i in range(len(df['features'])): #For each instance
          try:
            featurescore.append(scores[i][features[i].index(feature)]) #Add the score for the feature we are looking for to the list
          except Exception as w:
            if (verbose):
              print("Warning: " + feature + " not found for instance " + str(i) + " in file " + filename + ". Assuming zero importance for that specific instance.")
            featurescore.append(0.0)
            
        data[explainer] =  featurescore #Set a column of the dataframe to the list of scores for the feature given a particular explainer
        
    except Exception as e:
        print("An error occurred while analyzing the graph. " + str(e)) #Something went wrong

  #print("Correlation map for explanations:")
  #plt.matshow(data.corr())
  #plt.show()
  for key, value in addendumkwargs.items(): #For each additional list to be plotted
    data[key] = value #Add it to the dataframe
  data.plot(title = feature) #Plot the data
  data.to_csv(feature + " " + model + ' .csv', index = False) #Store the dataframe on a file
  return data.corr() if len(data.columns) != 2 else data.corr()[explainers[0]][explainers[1]] #Return the correlation if only 2 explainers are being compared
    
def maxImportanceScoreGenerator(filenames:list): #Generate the maxScores addendum list
    maxdf = pd.DataFrame()
    for filename in filenames:
        df = pd.read_csv(filename)
        explainer = filename.split()[0]
        maxScore = []
        maxScoreFeature = []
        for i in range(len(df["scores"])):
            maxScore.append(max(df["scores"][i]))
            maxScoreFeature.append(df["features"][i][df["scores"][i].index(max(df["scores"][i]))])
        
        maxdf[explainer + " maxScore"] = maxScore
        maxdf[explainer + " maxScoreFeature"] = maxScoreFeature
    return maxdf
        
        
