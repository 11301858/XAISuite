from .imports import*

def compare_explanations(filenames:list, verbose = False, **addendumkwargs): #Analyze the generated explanations in given files
  '''A function that analyzes and compares the explanations generated by train_and_explainModel.
    
    :param list filenames: File names with explanations (of the form "Explainer ImportanceScores - Model Target.csv")
    :param ``**addendumkwargs``: Any additional columns to be added to analysis. Each new parameter should be of the form addendumName = [addendumList]]
    :return: None
    '''
  dataset = addendumkwargs["dataset"] if "dataset" in addendumkwargs else ""
  if "dataset" in addendumkwargs:
    del addendumkwargs["dataset"]
    
  print("There are " + str(len(filenames)) + " files.")
  corrList = []
  model = filenames[0].split()[3] #All files should have the same model
  try:
    df = pd.read_csv(filenames[0])
    df['features'][0] = ast.literal_eval(df['features'][0])
    for feature in df['features'][0]:
        if len(filenames) != 2:
          compare_explanationssinglef(filenames, feature, verbose, **addendumkwargs)
        else:
           corrList.append(compare_explanationssinglef(filenames, feature, verbose, **addendumkwargs))
    
    if len(filenames) == 2:
      import statistics
      print("Average correlation is " + str(statistics.fmean(corrList)))
    
    if len(filenames) == 2:
      print("Printing in-depth information since 2 explainers are provided.")
      try:
        data = pd.read_csv("featuresvsmodel" + dataset + ".csv")
      except Exception as e:
        with open("featuresvsmodel" + dataset + ".csv", 'w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(["Model"] + df['features'][0])
        data = pd.read_csv("featuresvsmodel" + dataset + ".csv")
      finally:
        data.loc[len(data.index)] = [model] + corrList
        print("List of correlations is \n" + str(data.head()))
        print("Correlation map for different features with given model between " + filenames[0].split()[0] + " and " + filenames[1].split()[0])
        data.set_index('Model', inplace=True, drop=True)
        plt.xticks(labels = df['features'][0])
        plt.yticks(labels = data.index)
        plt.matshow(data)
        
        plt.show()
        data.to_csv("featuresvsmodel" + dataset + ".csv")
        
      
  except Exception as e:
    print("An error occurred while analyzing the graph. " + str(e))
  

def compare_explanationssinglef(filenames:list, feature:str, verbose = False, **addendumkwargs): #Analyze the generated explanations in given files
  '''A function that analyzes and compares the explanations generated by train_and_explainModel.
    
    :param list filenames: File names with explanations (of the form "Explainer ImportanceScores - Model Target.csv")
    :param str feature: Feature whose importance scores are to be compared
    :param ``**addendumkwargs``: Any additional columns to be added to analysis. Each new parameter should be of the form addendumName = [addendumList]]
    :return: Correlation between different explainer files
    '''
  explainers = []
  features = [] #All files should have same features
  model = filenames[0].split()[3] #All files should have the same model
  data = pd.DataFrame()
  for filename in filenames:
    try: 
        df = pd.read_csv(filename)

        for i in range(len(df['features'])):
            if(verbose):
                print (df['features'][i])
            df['features'][i] = ast.literal_eval(df['features'][i])
            #df.loc[:, ('features', i)] = ast.literal_eval(df['features'][i])

        for i in range(len(df['scores'])):
            if(verbose):
                print (df['features'][i])
            df['scores'][i] = ast.literal_eval(df['scores'][i])
            #df.loc[:, ('scores', i)] = ast.literal_eval(df['scores'][i])

        features = df['features']
        scores = df['scores']
        explainer = filename.split()[0]
        explainers.append(explainer)

        
        vars()[feature.replace(' ', '').replace('(', '').replace(')', '') + explainer + "List"] = []
        for i in range(len(df['features'])):
            eval(feature.replace(' ', '').replace('(', '').replace(')', '') + explainer + "List").append(scores[i][features[i].index(feature)])
        
        data[explainer] =  eval(feature.replace(' ', '').replace('(', '').replace(')', '') + explainer + "List")   
        
    except Exception as e:
        print("An error occurred while analyzing the graph. " + str(e))

  #print("Correlation map for explanations:")
  #plt.matshow(data.corr())
  #plt.show()
  for key, value in addendumkwargs.items():
    data[key] = value
  data.plot(title = feature)
  data.to_csv(feature + " " + model + ' .csv')
  return data.corr() if len(filenames) != 2 else data.corr()[explainers[0]][explainers[1]]
    
def maxImportanceScoreGenerator(filenames:list): #Generate the maxScores addendum list
    for filename in filenames:
        df = pd.read_csv(filename)
        explainer = filename.split()[0]
        vars()[explainer + "maxScore"] = []
        vars()[explainer + "maxScoreFeature"] = []
        for i in range(len(df["scores"])):
            eval(explainer + "maxScore").append(max(df["scores"][i]))
            eval(explainer + "maxScoreFeature").append(df["features"][i][df["scores"][i].index(max(df["scores"][i]))])
        
        return eval(explainer + "maxScore"), eval(explainer + "maxScoreFeature")
        
        
