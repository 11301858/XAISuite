from .imports import* #Import necessary libraries

#Explandable data loder. Users can propose functions to load data from a variety of places. As of now, the data loading harbor contains two piers: one for CSV file and one for sk-learn premade datasets.


def load_data_CSV(data:str, target:str, cut: Union[str, list] = None) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular dataset.
    
    :param str data: Pathname for the CSV file where the dataset is found.
    :param str target: Target variable used for training data
    :param Union[str, list] cut: Variables that should be ignored in training. By default, None.
    :return: Tabular object instance representing 'data'
    :rtype: Tabular
    :raises ValueError: if data cannot be loaded
    '''
    try:
        df = pd.read_csv(data) #We read the dataset onto a dataframes object
        if cut is not None:
          df.drop(cut, axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
          print(df)
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object

def load_data_sklearn(datastore:dict, target:str, cut: Union[str, list] = None) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular sklearn dataset for demoing.
 
    :param dict datastore: A dictionary object containing the data
    :param str target: Target variable used for training data
    :param Union[str, list] cut: Variables that should be ignored in training. By default, None
    :return: Tabular object instance representing 'data'
    :rtype: Tabular
    :raises ValueError: if data cannot be loaded
    '''
    try:
        df = pd.DataFrame(data=datastore.data, columns=datastore.feature_names) #We read the dataset (exluding target) onto a dataframes object
        df["target"] = datastore.target #We add the target column
        if cut is not None:
          df.drop([cut], axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object
        
def generate_data(type:str, target:str, cut: Union[int, list] = None, **generationArgs) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular sklearn dataset for demoing.
 
    :param type str: Type of data to generate ("classification" or "regression")
    :param target str: name of target variable
    :param ``**generationArgs``: Arguments to be passed onto the data generation function
    :return: Tabular object instance representing randomly generated dataset
    :rtype: Tabular
    :raises ValueError: if data cannot be loaded
    '''
    df = pd.DataFrame()
    try:
        if (type == "classification"):
            data = make_classification(**generationArgs)
            df = pd.DataFrame(data[0])
            df["target"] = data[1]
        elif (type == "regression"):
            data = make_regression(**generationArgs)
            df = pd.DataFrame(data[0])
            df["target"] = data[1]
        else:
            raise ValueError("Not correct type. Choices are 'classification' or 'regression'.")
          
        if cut is not None:
          df.drop([cut], axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object

