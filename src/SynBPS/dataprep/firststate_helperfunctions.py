
def fs_prepare_dataset_from_memory(Input_data, sample=1.0, transform="log", first_state=True, verbose=True):
    import pandas as pd
    import numpy as np
    
    def generate_first_state_table(X, y):
        """
        First-state encoding
        """
        # First timestep:
        X = X[:,:1,:]
        X.shape
        
        # Reshape to 2D
        X = X.reshape([X.shape[0],X.shape[2]])
        print(X.shape)
        
        # Feature table
        X = pd.DataFrame(X)
        
        # Reshape to a vector
        y.shape
        y = y.reshape(y.shape[0])
        print(y.shape)
        
        # Add targets to the feature table
        X["y_time_to_ev"] = y +1 # add 1 so that weibull model will work
        X["y_closed"] = 1
        
        # Logic for dropping features
        numcols = len(X.columns)
        
        # Dropping all features but the first, and preserving three last
        drops = range(1,numcols-2)
        X = X.drop(drops,axis=1)
        
        #print(X)        
        return X

    ## Load the numpy files
    x_train = Input_data["x_train"]
    x_test = Input_data["x_test"]
    y_train = Input_data["y_train"]
    y_test = Input_data["y_test"]
    y_t_train = Input_data["y_t_train"]
    y_t_test = Input_data["y_t_test"]
    
    ## load inference tables for evaluation and analysis
    Inference_train = Input_data["Inference_train"]
    Inference_test = Input_data["Inference_test"]
    
    ## Event-log characteristics
    n = x_train.shape[0]
    maxlen = x_train.shape[1]
    num_features = x_train.shape[2]  
    

    ## Sub-sampling the data:
    if sample < 1.0:
        
        #Get new size
        newsize = int(np.round(n*sample,decimals=0))
        
        #Subset based on index
        selected_ids = np.random.randint(n, size=newsize)
        x_train, y_train = x_train[selected_ids,:], y_train[selected_ids,:]

    ## Return the main objects as a dictionary
    data_objects = {"x_train":x_train, 
                    "x_test": x_test,
                    "y_train":y_train,
                    "y_test":y_test,
                    "y_t_train":y_t_train,
                    "y_t_test":y_t_test,
                    "maxlen":maxlen,
                    "num_features":num_features,
                    "Inference_train":Inference_train,
                    "Inference_test":Inference_test}
    
    
    #Make sure it has enogh precision
    y_train = np.float32(y_train)
    y_test = np.float32(y_test)
      
       
    # Transformations:
    if transform=="range":
        
        # Store the original values:
        data_objects["y_train_min"] = np.min(y_train)
        data_objects["y_train_max"] = np.max(y_train)
        data_objects["y_test_min"] = np.min(y_test)
        data_objects["y_test_max"] = np.max(y_test)
        
        # Train data
        y_train = (y_train -  data_objects["y_train_min"])/(data_objects["y_train_max"] - data_objects["y_train_min"])
        y_test = (y_test -  data_objects["y_test_min"])/(data_objects["y_test_max"] - data_objects["y_test_min"])
        
        
        #Inverse range transform
        #Inference_test["y_pred"] = (Inference_test["y_pred"] * (data_objects["y_test_max"] - data_objects["y_test_min"])) + data_objects["y_test_min"]
    
                
    if transform=="log":
        #Log-transform
        y_train = np.log(1+y_train)
        y_test = np.log(1+y_test)    
        
        # Inference data
        #Inference_test["y"] = np.log(1+Inference_test["y"])
        #Inference_test["y_t"] = np.log(1+Inference_test["y_t"])
        
        #Inverse log transform
        #Inference_test["y_pred"] = np.exp(#Inference_test["y_pred"])-1
    
    # Store transformed target
    data_objects["y_train"] = np.float32(y_train)
    data_objects["y_test"] = np.float32(y_test)    
    
    data_objects["Inference_train"] = Inference_train
    data_objects["Inference_test"] = Inference_test
    
    # Store first-state version of the data
    if first_state == True:
        data_objects["xy_train_firststate"] = generate_first_state_table(x_train, data_objects["y_train"])
        data_objects["xy_test_firststate"] = generate_first_state_table(x_test, data_objects["y_test"])
    
    return data_objects

