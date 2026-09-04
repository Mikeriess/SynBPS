def make_D(statespace):
    D=[]#list(range(1, statespace + 1))
    D = ["S"+str(s) for s in D]
    
    alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","w","x","y","z","æ","ø","å"]
    
    for i in range(1, statespace + 1):
        D.append(alphabet[i])
    
    return D

def make_workweek(workweek):
    #error handling
    if workweek not in ["weekdays", "all-week", "none"]:
        raise ValueError("Deterministic_offset_W must be weekdays, all-week or none, but is " + str(workweek) + ". Change Deterministic_offset_W in the settings.")
    
    if workweek == "weekdays":
        # CLOSED HOURS FROM
        W = [[0.0, #monday
             1, #tuesday
             2, #wednesday
             3, #thursday
             4, #friday
             5],
            # TO
            [0.5, #monday
             1.5, #tuesday
             2.5, #wednesday
             3.5, #thursday
             4.5, #friday
             7.5]]  #weekend-closed

    if workweek == "all-week":
        # CLOSED HOURS FROM
        W = [[0.0,
             1, 
             2, 
             3,
             4, 
             5,
             6,
             7],
            # TO
            [0.5, #monday
             1.5, #tuesday
             2.5, #wednesday
             3.5, #thursday
             4.5, #friday
             5.5, #saturday
             6.5, #sunday
             7.5]] 
    
    if workweek == "none":
        # no closed hours: an empty list of intervals, so the deterministic offset is always 0
        W = [[],
             []]
    
    return W

def flatten(listoflists):    
    flattened_list = [item for sublist in listoflists for item in sublist]
    return flattened_list