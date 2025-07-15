import numpy as np 
import pandas as pd 
import itertools 
import random


def update_variable_factor(row, column):
    # Pronadji dolazece poruke od faktora prema datoj varijabli
    messages = A.iloc[:,column].values
    messages_idx = list(np.nonzero(messages)[0])
    # Uzmi sve poruke osim one prema kojoj saljemo update
    index = messages_idx.index(row)
    incoming_messages = messages_idx[:index] + messages_idx[index+1:]
    
    # Prodji kroz razlicite vrednost
    for v in values:
        message_product = 1
        # Odradi proizvod poruka
        for incoming_message in incoming_messages:
            message_product *= M_factor_variable[incoming_message, column, v].copy()
        # Azuriraj vrednosti
        M_variable_factor[row, column, v] += np.exp(c[column] * v) * message_product


def update_factor_variable(row, column):
    messages = A.iloc[row].values
    messages_idx = list(np.nonzero(messages)[0])
    index = messages_idx.index(column)
    incoming_messages = messages_idx[:index] + messages_idx[index+1:]

    for v in values:
        # Mnozimo sve dolazece poruke sa variable cvorova 
        message_product = 1
        for incoming_message in incoming_messages:
            # Dolazece poruke za dato ogranicenje
            message_product *= M_variable_factor[row, incoming_message, v].copy()

        combinations = list(itertools.product(values, repeat = len(incoming_messages)+1))
        combinations = list(filter(lambda x: x[index] == v,combinations))

        for e in combinations:
            psi = psi_dict[f'psi_{row + 1}'].loc[e].values.copy()
            M_factor_variable[row, column, v] += psi * message_product
    

def calculate_variable_beliefs():
    for column in range(M.shape[1]):
        messages = A.iloc[:,column].values
        # Uzimam ne nula elemente
        messages_idx = list(np.nonzero(messages)[0])
        # Idemo po nenula elementima
        for v in values:
            message_product = 1
            for incoming_message in messages_idx:
                message_product *= M_factor_variable[incoming_message, column, v]

            variable_beliefs[v, column] = np.exp(c[column] * v) * message_product
            
    display(variable_beliefs)
    print('Solution for vector x: ',np.argmax(variable_beliefs,axis=0))

def calculate_factor_belief():
    for row in range(M.shape[0]):
        # Idem po prvom redu
        messages = A.iloc[row].values
        #print(messages)

        # Uzimam ne nula elemente
        messages_idx = list(np.nonzero(messages)[0])
        #print('m_idx',messages_idx)
        # Idemo po nenula elementima
        for idx_j, j in enumerate(messages_idx):
            for v in values:
                # Mnozimo sve dolazece poruke sa variable cvorova 
                message_product = 1
                for incoming_message in messages_idx:
                    # Dolazece poruke za dato ogranicenje
                    message_product *= M_variable_factor[row, incoming_message, v].copy()

                #print('Message product', message_product)
                #print('Trenutna vrendost za v: ',v)
                #print('Saljemo poruku ka kojoj varijabli tj. koja nam je fiksirana v: ', j)
                combinations = list(itertools.product(values, repeat = len(messages_idx)))
                #print('Full combinations:', combinations)
                combinations = list(filter(lambda x: x[idx_j] == v,combinations))
                #print(combinations)
                #display(factor_belief[f'psi_{row + 1}'])
                for e in combinations:
                    psi = psi_dict[f'psi_{row + 1}'].loc[e].values.copy()
                    factor_belief[f'belief_{row + 1}'].loc[e] =  psi * message_product


if __name__ == '__main__':
    num_of_variables = 3 
    columns = list(range(1,num_of_variables+1))
    A = pd.DataFrame([[1,1,0],
                [1,0,1],
                [0,1,1],
                [1,1,1]],columns = columns)
    d = pd.Series(data = [1,1,1,1])
    c = np.array([2,1,1])
    values = [0,1]
    variable_beliefs = np.zeros((len(values),len(c)))

    psi_dict = {}
    factor_belief = {}
    for idx,row in A.iterrows():
        # For every constraint take variables that are figuring in it 
        columns = (list(A.columns[row == 1]))
        # Make combination for combination of different values for different variables
        combinations = list(itertools.product(values,repeat=len(columns)))
        
        # Create factors so that each factor sees when to we satisfy limitations
        df = pd.DataFrame(combinations, columns=columns)
        psi_values = df.dot(A.loc[idx,columns]) <= d[idx]
        df['psi'] =  psi_values
        psi_dict[f'psi_{idx+1}'] = df.set_index(list(df.columns[:-1]))
        factor_belief[f'belief_{idx+1}'] = df.set_index(list(df.columns[:-1]))
        print(f'psi_{idx+1}')
        display(psi_dict[f'psi_{idx+1}'])


    M = np.dstack([A.copy(), A.copy()])

    M_factor_variable = np.random.rand(*M.shape) * M
    M_variable_factor = np.random.rand(*M.shape) * M

    M_factor_variable, M_variable_factor = M.copy(), M.copy()
    num_iterations = 7

    # Pronajdi sve grane i nasumicno ih poredjaj
    update_list = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A.iloc[i,j]:
                update_list.append(f'fv{i}{j}')
                update_list.append(f'vf{i}{j}')
    random.shuffle(update_list)

    for _ in range(1):
        for update in update_list:
            # Extract edges
            row = int(update[2])
            column = int(update[3])
            direction = update[:2]

            if direction == 'fv':
                update_factor_variable(row,column)
            elif direction == 'vf':
                update_variable_factor(row, column)
        

    calculate_variable_beliefs()