import pandas as pd
import os 
import csv
import random
import time
import numpy as np
from threading import *
import threading
import matplotlib.pyplot as plt



directory = "Before_MergeData"
parent = os.getcwd()
path = os.path.join(parent, directory)

try:
    os.makedirs(path, exist_ok = True)
    print("Directory '%s' created  successfully" % directory)
except OSError as error:
    print("Directory '%s' already exists " % directory)

def create_directory(directory_name, parent):
    val = os.path.join(parent, directory_name)
    try:
        os.makedirs(val, exist_ok = True)
        print("Directory '%s' created  successfully" % directory_name)
    except OSError as error:
        print("Directory '%s' already exists " % directory_name)
    return val
    
def create_file(f_name):
    with open (f_name, 'w') as csv_fl:
        print("file created successfully")
        return f_name
    csv_fl.close()

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)



from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time
##Threading:
##- A new thread is spawned within the existing process
##- starting a thread is faster than starting a process
##- memory is shared between all threads
##- mutexes often necessary to control access to shared data
##- on GIL (Global Interpreter Lock) for all threads
##semaphore locks with lock access count
semTrn = Semaphore(4)
semLd = Semaphore(4)

def extract(file):
    dtype_dict = {'Name': 'object',
                  'Gender': 'object',
                  'Salary': 'float64',
                  'empid': 'float64',
                  'Address': 'object'
                  }

    df = pd.read_csv(file, dtype=dtype_dict, low_memory=False)
    return df


def transform(df):
    ##semaphore lock
    semTrn.acquire()
    print("thread {} acquired tranform lock ".format(threading.currentThread().ident))
    ##basic transformation operation

    
    name_df2 = df.Name.str.split(" ",expand=True)
    df.drop(['Name'], axis=1,inplace=True)
    df.insert(0,'First name',name_df2.iloc[:,0])    
    df.insert(1,'Last name',name_df2.iloc[:,1])
    df.loc[df["Gender"] == "Male", "Gender"] = 'M'
    df.loc[df["Gender"] == "Female", "Gender"] = 'F'
    df.rename(columns = {'Gender':'Sex'}, inplace = True)
    df.rename(columns = {'empid':'id'}, inplace = True)
    semTrn.release()
    print("thread {} released tranform lock ".format(threading.currentThread().ident))
    print("thread {} acquired load lock ".format(threading.currentThread().ident))
    semLd.acquire()
    load(df)


def load(tdf):

    tdf.to_csv('Threadedfile.csv', mode='a', header=False, index=False)
    semLd.release()
    print("thread {} released load lock  ".format(threading.currentThread().ident))
    print("thread {} load completion ".format(threading.currentThread().ident))

#Name Gender Salary empid Address

def transformnappend(file_path_2, indices, database):

    to_append = pd.DataFrame()
    to_append['First name'] = []
    to_append['Last name'] = []
    to_append['Sex'] = []
    to_append['Salary'] = []
    to_append['id'] = []
    to_append['Address'] = []

    # print(indices)
    for x,y in pairwise(indices):
        # to_transform = database.loc[x:y]
        # print(to_transform.head())
        # # to_transform[['First name','Last name']] = to_transform.Name.str.split(" ",expand=True)
        # #to_transform.loc['First name':'Last name'] = to_transform.Name.str.split(" ",expand=True)
        # #to_transform.drop(['Name'], axis=1)
        # to_transform.loc[to_transform["Gender"] == "Male", "Gender"] = 'M'
        # to_transform.loc[to_transform["Gender"] == "Female", "Gender"] = 'F'
        # to_transform.rename(columns = {'Gender':'Sex'}, inplace = True)
        # pd.concat([to_append,to_transform], ignore_index = True)
         to_transform = database.loc[x:y]
         name_df=to_transform.Name.str.split(" ",expand=True)
         to_transform.drop(['Name'], axis=1,inplace=True)
         to_transform.insert(0,'First name',name_df.iloc[:,0])    
         to_transform.insert(1,'Last name',name_df.iloc[:,1])
         print(to_transform.head())
   
         to_transform.loc[to_transform["Gender"] == "Male", "Gender"] = 'M'
         to_transform.loc[to_transform["Gender"] == "Female", "Gender"] = 'F'
         to_transform.rename(columns = {'Gender':'Sex'}, inplace = True)
         to_transform.rename(columns = {'empid':'id'}, inplace = True)
         to_append=pd.concat([to_append,to_transform], ignore_index = True)
         
    print(to_append.dtypes)
    to_append.to_csv(file_path_2)





#create a pandas dataframe of 10 L records and write it to a csv file
#create file
#if file does not exist create and add 10 L records
file = "Employee_data.csv"
flag = os.path.exists(os.path.join(path, file))
file_path = os.path.join(path, file)
if(flag):
    print("The file already exists")
else:
    fields = ['Name','Gender','Salary','empid','Address']
    # atleast while writing names no need for dataframe
    rows = []  # Empty 2d list 

    first_names = ['Rahul','Gautam','Prashant','Nimit','Srijan','Kinjal','Aayush']
    last_names = ['Kaushal','Jain','Bhalla','Dhawan','Gupta','Goyal','Asrey']
    Gender = ['Male','Female']
    Address= ['Patiala','Noida','Mohali','Gurgaon','Sunam','Delhi']

    for x in range(1000000):
        emp_id = str(x+1)
        name1 = random.choice(first_names)
        name2 = random.choice(last_names)
        sp = " "
        name = name1 + sp + name2 
        gender = random.choice(Gender)
        address = random.choice(Address)
        salary = str(random.randrange(1000000,2000000))
        entries = [name, gender, salary, emp_id, address]
        rows.append(entries)

    with open(file_path, 'w',newline = "") as csvfile:#csvfile is the name of the file object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
        print("File created successfully")
    csvfile.close()
#Operation 1 transferring data converting one record at a time
#Operation 2 multiple files at a time 
#Operation 3 multiple files at a time using multithreading
#two options for reading the file pandas_readcsv or opening the file in read mode and then writing using the csv module
#Operation 1
#start1 = time.time()
#df_1 = pd.read_csv(file_path)
#print(len(df_1)) checking if rows are correct in number or not
c_1 = create_directory("Case_1",parent)
c_2 = create_directory("Case_2",parent)
c_3 = create_directory("Case_3",parent)
c_list = [c_1, c_2, c_3]
f_list = ["Onebyone.csv","Chunkfiles.csv","Pipelining.csv"]
f_path_list = []

for path_y,file_z in list(zip(c_list,f_list)):
    to_create = os.path.join(path_y,file_z)
    if(os.path.exists(to_create)):
        print("the file already exists")
        f_path_list.append(to_create)
    else:
        f_path_list.append(create_file(to_create))

#print(f_path_list)
#pick the files one by one and perform the operations on them
#Now for the data conversion fetch a row from the csv and convert it into required format
#First we must open the file and pass the entire data to a function which converts our data into the required format

#csvwriterows writes multiple rows
with open(file_path, 'r', newline='') as work_file:
    reader = csv.reader(work_file,delimiter=',')
    with open(f_path_list[0], 'a',newline = '') as update_file:
        writer_obj = csv.writer(update_file)
        temp_data1 = ["First name","Last name", "Sex", "Salary","id","Address"]
        writer_obj.writerow(temp_data1)
        for d_entry in reader:
            
            if(d_entry[0] == "Name"):
                continue
            else:
                temp_data2 = []

                x = d_entry[0].split(" ")
                temp_data2.append(x[0])
                temp_data2.append(x[1])

                if(d_entry[1] == "Female"):
                    temp_data2.append("F")
                else:
                   temp_data2.append("M")

                temp_data2.append(d_entry[2])
                temp_data2.append(d_entry[3])
                temp_data2.append(d_entry[4])
                writer_obj.writerow(temp_data2)

        update_file.close()
    work_file.close()

df = pd.read_csv(file_path)

#df_2 = df_1[1:500000]
#df_3 = df_2[500001:1000000]
splits = [2,4,5,8,10]
ind_array_or = []
for split_value in splits:
    ind_array = []
    start_index = 0
    end_index = (1000000)/split_value - 1
    increment = end_index 
    ind_array.append(int(start_index))
    ind_array.append(int(end_index))
    for x in range(split_value-1):
        start_index = end_index + 1
        end_index = start_index + increment 
        ind_array.append(int(start_index))
        ind_array.append(int(end_index))
    ind_array_or.append(ind_array)

print(ind_array_or)
    #read the csv using pandas
large_data = pd.read_csv(file_path)
    
file_names_2 = ["5Lrecords.csv","2.5Lrecords.csv","2Lrecords.csv","1.25Lrecords.csv","1Lrecords.csv"]

pass_ind = 0
path_Case2 = "Case_2"
time_taken = []



for file_2 in file_names_2:

    pass_list = ind_array_or[pass_ind]
    check_file = os.path.join(path_Case2, file_2)
    pass_ind +=1
    if(os.path.exists(check_file)):
        print("The file already exists")
    else:
        created_file = create_file(check_file)
        timer_start = time.time()  # calculate the time for transform and append function
        transformnappend(created_file,pass_list,large_data)
        timer_end = time.time()
        time_reqd = (timer_end - timer_start)
        print(time_reqd)
        time_taken.append(time_reqd)
time_values = pd.DataFrame()
time_values['Records in file'] = ['1L','1.25L','2L','2.5L','5L']
time_taken = time_taken[::-1]
time_values['Time required'] = time_taken
print(time_values.head())
f_name = 'Comparison.csv'
comp_path = os.path.join(parent, f_name)
if(os.path.exists(comp_path)):
        print("The file already exists")
else:
    time_values.to_csv(comp_path)    



#Read the file and convert the required column into a numpy array
plot_data = pd.read_csv(comp_path)
y = plot_data['Time required']
y = y.to_numpy()
x = [1,1.25,2,2.5,5]
plt.plot(x, y)
plt.title("Curve plotted using the given points")
plt.xlabel("File Size")
plt.ylabel("Time Required")
#plt.show()
        
# threaded function comes here
#File with final multithreading and we find that 2.5 was the optimal solution, values may change on each run so values are saved in form of png

def main1():
    pd.set_option('mode.chained_assignment', None)
    file = file_path
    df = extract(file)
    chunk_size = int(df.shape[0] / 4)
    ##t = [0] * 4
    executor = ThreadPoolExecutor(max_workers=4)
    lst = list()
    for start in range(0, df.shape[0], chunk_size):
        df_subset = df.iloc[start:start + chunk_size]
        ##df_subset.is_copy=None
        lst.append(executor.submit(transform, df_subset))
    for future in lst:
        future.result()
    executor.shutdown()


start = time.time()
main1()
end = time.time() - start
print("Execution time {} sec".format(end))



    


        







    

            


            


            





































