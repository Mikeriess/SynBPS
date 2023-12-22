def make_D(statespace):
    D=[]#list(range(1, statespace + 1))
    D = ["S"+str(s) for s in D]
    
    alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","w","x","y","z","æ","ø","å"]
    
    for i in range(1, statespace + 1):
        D.append(alphabet[i])
    
    return D

def make_workweek(workweek):
    if workweek == "weekdays":
        # CLOSED HOURS FROM
        W = [[0.001, #monday
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
        W = [[0.001,
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
    return W

def flatten(listoflists):    
    flattened_list = [item for sublist in listoflists for item in sublist]
    return flattened_list