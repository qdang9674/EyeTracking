import mat73
import pandas as pd

data_dict = mat73.loadmat('/home/manoruo1/jbrook1_ada/users/manoruo1/NewPup_withGazeData.mat')


def getSubjectDataFromDataBlock(datablock):
    """
        Returns a dictionary structured the followig way:

        subject_index (there are 58 subjects):
            dataCategories (i.e 'Behavior', 'eyeposX', etc.):
                trials (each category should have 10 trials):
                    trialData (the actual data)
    """
    subject_data_dict = {}
    for key in datablock.keys():
        subjects_data = datablock[key]  # list

        for i in range(len(subjects_data)):

            subject_data = subjects_data[i]  # get the subjects data
            subject_name = "subject" + str(i + 1)

            if not subject_name in subject_data_dict:
                subject_data_dict[subject_name] = {}

            if not key in subject_data_dict[subject_name]:
                subject_data_dict[subject_name][key] = {}

            for trial in range(len(subject_data)):
                trial_data = subject_data[trial]
                subject_data_dict[subject_name][key]["trial" + str(trial)] = trial_data
    return subject_data_dict


def GetExperiementDataFrames(block_data_dict, taskType):
    eye_data_df = pd.DataFrame(columns=[ 'Task Type','SID', "Trial", 'Time', 'Pupil Diameter', "Pupil X", "Pupil Y"])
    behavioral_data_df = pd.DataFrame(columns=['Task Type', 'SID', 'Trial', 'Stimulus Time', "Reaction Time",
                                               "isCorrectResponse"])

    for subject_index, subjectID in enumerate(block_data_dict.keys()):
        #create 2 inner dataFrame to reduce the growth rate
        eye_data_df_inner = pd.DataFrame(columns=[ 'Task Type','SID', "Trial", 'Time', 'Pupil Diameter', "Pupil X", "Pupil Y"])
        behavioral_data_df_inner = pd.DataFrame(columns=['Task Type', 'SID', 'Trial', 'Stimulus Time', "Reaction Time",
                                               "isCorrectResponse"])
        
        print("At subject: ", subject_index)
        data = block_data_dict[subjectID]

        behave_data = data["Behavior"]
        pupil_diam = data["pupil_diam"]
        time_stamp = data["time_ms"]
        eyeposx = data["eyeposX"]
        eyeposy = data["eyeposY"]

        MAX_NUM_TRIALS = 10  # each should have 10 trials

        for i in range(MAX_NUM_TRIALS):
            #print("At trial: ", i)
            trial = "trial" + str(i)

            curr_behav = behave_data[trial]
            curr_pupil_diam = pupil_diam[trial]
            curr_time_stamp = time_stamp[trial]
            curr_eyepos_x = eyeposx[trial]
            curr_eyepos_y = eyeposy[trial]

            if (curr_pupil_diam is not None and curr_time_stamp is not None and curr_eyepos_x is not None and curr_eyepos_y is not None and curr_behav is not None):

                # make eye data df
                # The pupil diameter, time stamp, eyeposx and eyeposy are all the same size (we will just use curr_pupil_diam as default size)
#                 for i in range(curr_pupil_diam.size):
#                     # get corresponding/related values at the current index
#                     pupil_diam_val = curr_pupil_diam[i]
#                     time_stamp_val = curr_time_stamp[i]  #this is in milisecond convert to seconds
#                     eyepos_x_val = curr_eyepos_x[i]
#                     eyepos_y_val = curr_eyepos_y[i]

#                     eye_data_df.loc[len(eye_data_df.index)] = [subjectID, taskType, time_stamp_val, pupil_diam_val,
#                                                                eyepos_x_val, eyepos_y_val, trial]
                curr_data_df= pd.DataFrame(columns=['SID', 'Task Type', 'Time', 'Pupil Diameter', "Pupil X", "Pupil Y", "Trial"])
                curr_data_df['Time'] = curr_time_stamp
                curr_data_df['Pupil Diameter'] = curr_pupil_diam
                curr_data_df['Pupil X'] = curr_eyepos_x
                curr_data_df['Pupil Y'] = curr_eyepos_y
                curr_data_df['SID'] = subject_index
                curr_data_df['Task Type'] = taskType
                curr_data_df['Trial'] = i
                eye_data_df_inner = pd.concat([eye_data_df_inner,curr_data_df],ignore_index=True)
                
#                 # make behavioral df (all the list in this dictionary should be same size)
#                 for i in range(len(curr_behav["StimOnsetTime"])):
#                     stimTime_val = curr_behav["StimOnsetTime"][i]
#                     if taskType == "PVT":
#                         # flip this to get correct value (In Dr. Brooks notes)
#                         is_correct_val = 1 - curr_behav["isLapse"][i]
#                     else:
#                         is_correct_val = curr_behav["isCorrectResponse"][i]
#                     react_time_val = curr_behav["ReactionTime"][i]
#                     behavioral_data_df.loc[len(behavioral_data_df.index)] = [str(subjectID), taskType,
#                                                                              stimTime_val, react_time_val,
#                                                                              is_correct_val, trial]
                    
                curr_behave_df = pd.DataFrame(columns=['SID', 'Task Type', 'Stimulus Time', "Reaction Time",
                                               "isCorrectResponse", "Trial"])
                
                curr_behave_df['Stimulus Time'] = curr_behav["StimOnsetTime"]
                curr_behave_df['Reaction Time'] = curr_behav["ReactionTime"]
                if taskType == "PVT":
                    # flip this to get correct value (In Dr. Brooks notes)
                    curr_behave_df["isCorrectResponse"] = 1 - curr_behav["isLapse"]
                else:
                    curr_behave_df["isCorrectResponse"] = curr_behav["isCorrectResponse"]
                curr_behave_df['SID'] = subject_index
                curr_behave_df['Task Type'] = taskType
                curr_behave_df['Trial'] = i
                
                behavioral_data_df_inner= pd.concat([behavioral_data_df_inner,curr_behave_df],ignore_index=True)
            else:
                #print("At else")
                # data may have not been recorded for this trial
                #print("No data for", trial)
                continue
        #end of inner loop
        
        eye_data_df = pd.concat([eye_data_df ,eye_data_df_inner],ignore_index=True)
        behavioral_data_df= pd.concat([behavioral_data_df ,behavioral_data_df_inner],ignore_index=True)
    #end of outer loop
        
    eye_data_df['Time'] = eye_data_df['Time'] * 0.001  #This is in milisecond convert to seconds                    
    behavioral_data_df["isCorrectResponse"] = behavioral_data_df["isCorrectResponse"]*1 #This will change the type for bool to int            
    return eye_data_df, behavioral_data_df


if __name__ == "__main__":

    data = data_dict["NewPup"]  # this is the key for all the data
    # all the data folders in our experiements are ['DPT', 'DYN', 'MA', 'PVT', 'REST', 'VWM'], we will focus on MA for now

    # to access data do [taskType]["block"][0]
    MA_data = getSubjectDataFromDataBlock(data["MA"]["block"][0])
    PVT_data = getSubjectDataFromDataBlock(data["PVT"]["block"][0])
    DPT_data = getSubjectDataFromDataBlock(data["DPT"]["block"][0])
    VWM_data = getSubjectDataFromDataBlock(data["VWM"]["block"][0])

    for taskType in data.keys():
        # loops through
        task_data = getSubjectDataFromDataBlock(data[taskType]["block"][0])  # gets organized dictionary with task data
        eye_data_df, behavioral_data_df = GetExperiementDataFrames(task_data, taskType)  # gets the actual dataframes

