#!/usr/bin/env python
# coding: utf-8

# In[1]:


## This code reads a openfoam simulation and creates a csv of state and flow fields at all co-ordinates and time

## BEFORE RUNNING THIS CODE ##
# 1. Go to controlDict and change writeFormat to ascii
# $ foamFormatConvert
# 2. Generate co-ordinates for all cells by running
# $ postProcess -func writeCellCentres


# In[2]:


# field_vals_at_time("wingMotion2D_pimpleFoam/0.26")


# In[3]:


def state_vals_at_time(filename):
    file_s = os.path.join(filename,"uniform","sixDoFRigidBodyMotionState")
    f_s = open(file_s,'r')
    l_s=f_s.readlines()
#     print(l_s)

    for line in l_s:
        line_elems=line.split()
#         print(line_elems)
        if len(line_elems)<1:
            continue
        if(line_elems[0]=="centreOfRotation"):
            cor = list(float(val) for val in line_elems[2:5])
        if(line_elems[0]=="orientation"):
            rotation_matrix = list(float(val) for val in line_elems[2:11])
        if(line_elems[0]=="velocity"):
            vel = list(float(val) for val in line_elems[2:5])
        if(line_elems[0]=="acceleration"):
            acc = list(float(val) for val in line_elems[2:5])
        if(line_elems[0]=="angularMomentum"):
            ang = list(float(val) for val in line_elems[2:5])
        if(line_elems[0]=="torque"):
            tor = list(float(val) for val in line_elems[2:5])

    dict_state = {"t":float(os.path.basename(filename)),"centreOfRotation_x":cor[0],"centreOfRotation_y":cor[1],"centreOfRotation_z":cor[2],"orientation_cos":rotation_matrix[0],"orientation_sin":rotation_matrix[3],"velocity_x":vel[0],"velocity_y":vel[1],"velocity_z":vel[2],"acceleration_x":acc[0],"acceleration_y":acc[1],"acceleration_z":acc[2],"angularMomentum_x":ang[0],"angularMomentum_y":ang[1],"angularMomentum_z":ang[2],"torque_x":tor[0],"torque_y":tor[1],"torque_z":tor[2]}

    return dict_state




# In[4]:


import csv
import os
savepath =  "" #location

try:
    os.remove(os.path.join(savepath,"field_at_special_points.csv"))
except:
    print("No special field file existed before")

keys = ['t','x','y','z','u','v','w','p']

with open(os.path.join(savepath,"field_at_special_points.csv"),'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    print("[INFO] Header written.")
    output_file.close()

def print_special_point(dict_point):
        keys = dict_point.keys()
        with open(os.path.join(savepath,"field_at_special_points.csv"),'a') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writerow(dict_point)
#             print("[INFO] appended special field point.")
            output_file.close()



# In[5]:


import os
def field_vals_at_time(file, ignore_z=True, onlyRefinedRegion=True):

    ############## VELOCITY ###############
    file_U = os.path.join(file,"U")
    f_U = open(file_U,'r')
    l_U=f_U.readlines()
    vel_at_time=[]
#     coordinates=[]
    startReading = False
#     p_at_time = []
    for line in l_U:
        if(line == ')\n'):
            startReading =False
            break
        if startReading:
#             dataVal = float(line.rstrip())
#             p_at_time.append(dataVal)
            l=(line[1:-2].split())
            u = float(l[0])
            v = float(l[1])
            w = float(l[2])
            if ignore_z:
                w=0
#             print("u: "+str(u)+"\t"+"v: "+str(v)+"\t"+"z: "+str(z)+"\t")
            u_v_w = {"u": u,"v": v,"w": w}
            vel_at_time.append(u_v_w)
        if(line=='(\n'):
            startReading = True


    ################ COORDINATES ##################

    file_C = os.path.join(file,"C")
    f_C = open(file_C,'r')
    l_C=f_C.readlines()
#     vel_at_times=[]
    coordinates=[]
    startReading = False
#     p_at_time = []
    for line in l_C:
        if(line == ')\n'):
            startReading =False
            break
        if startReading:
#             dataVal = float(line.rstrip())
#             p_at_time.append(dataVal)
            l=(line[1:-2].split())
            x = float(l[0])
            y = float(l[1])
            z = float(l[2])
            if ignore_z:
                z=0.125
#             print("u: "+str(u)+"\t"+"v: "+str(v)+"\t"+"z: "+str(z)+"\t")
            x_y_z = {"x": x,"y": y,"z": z}
            coordinates.append(x_y_z)
        if(line=='(\n'):
            startReading = True


    #################### PRESSURE ########################
    file_p = os.path.join(file,"p")
    f_p = open(file_p,'r')
    l_p=f_p.readlines()
#     vel_at_times=[]
#     coordinates=[]
    startReading = False
    p_at_time = []
    for line in l_p:
        if(line == ')\n'):
            startReading =False
        if startReading:
            dataVal = float(line.rstrip())
            p_at_time.append(dataVal)
#             l=(line[1:-2].split())
#             x = float(l[0])
#             y = float(l[1])
#             z = float(l[2])
#             if ignore_z:
#                 z=0.125
#             print("u: "+str(u)+"\t"+"v: "+str(v)+"\t"+"z: "+str(z)+"\t")
#             p_dict = {"p": p}
#             p_at_time.append(p_dict)
        if(line=='(\n'):
            startReading = True

#     print(len(p_at_time),len(vel_at_time), len(coordinates))
#     print(vel_at_time[-10:])

    time_snap=[]
    for index,val in enumerate(p_at_time):
        if (not(coordinates[index]["x"]>-2 and coordinates[index]["x"]<8 and coordinates[index]["y"]<2.5 and coordinates[index]["y"]>-2.5)) :
            if (onlyRefinedRegion):
                continue
        state = state_vals_at_time(file)
        if (coordinates[index]["x"]-state["centreOfRotation_x"]>-0.5 and coordinates[index]["x"]-state["centreOfRotation_x"]<1.5 and coordinates[index]["y"]-state["centreOfRotation_y"]>-0.5 and coordinates[index]["y"]-state["centreOfRotation_y"]<0.5) :
            dict_special_pt = {"t":float(os.path.basename(file)),"x":coordinates[index]["x"],"y":coordinates[index]["y"],"z":coordinates[index]["z"],"u":vel_at_time[index]["u"],"v":vel_at_time[index]["v"],"w":vel_at_time[index]["w"],"p":p_at_time[index]}
            print_special_point(dict_special_pt)
            continue

        dict_at_coordinate = {"t":float(os.path.basename(file)),"x":coordinates[index]["x"],"y":coordinates[index]["y"],"z":coordinates[index]["z"],"u":vel_at_time[index]["u"],"v":vel_at_time[index]["v"],"p":p_at_time[index]}###-state["velocity_y"]###,"w":vel_at_time[index]["w"],"p":p_at_time[index]}
        time_snap.append(dict_at_coordinate)

    return time_snap




# In[6]:


# state_vals_at_time("wingMotion2D_pimpleFoam/0.26")


# In[7]:


import csv
savepath =  "" #location
try:
    os.remove(os.path.join(savepath,"field_at_all_time_space.csv"))
except:
    print("No field file existed before")

try:
    os.remove(os.path.join(savepath,"state_at_all_times.csv"))
except:
    print("No state file existed before")

#     os.remove(os.path.join(savepath,"state_at_all_times.csv"))

keys = ['t','x','y','z','u','v','w','p']
with open(os.path.join(savepath,"field_at_all_time_space.csv"),'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
#     dict_writer.writerow(state_snap)
    print("[INFO] Header written.")
    output_file.close()

def save_field(time_snap):
    keys = time_snap[0].keys()
#     print(keys)
    with open(os.path.join(savepath,"field_at_all_time_space.csv"),'a') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
#         dict_writer.writeheader()
        for time_snap_entry in time_snap:
            dict_writer.writerow(time_snap_entry)
        print("[INFO] appended field for time step: "+ str(time_snap[0]["t"]))
        output_file.close()


keys = ['t', 'centreOfRotation_x', 'centreOfRotation_y', 'centreOfRotation_z', 'orientation_cos', 'orientation_sin', 'velocity_x', 'velocity_y', 'velocity_z', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'angularMomentum_x', 'angularMomentum_y', 'angularMomentum_z', 'torque_x', 'torque_y', 'torque_z']
with open(os.path.join(savepath,"state_at_all_times.csv"),'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
#     dict_writer.writerow(state_snap)
    print("[INFO] Header written.")
    output_file.close()


def save_state(state_snap):
    keys = state_snap.keys()
#     print(keys)
    with open(os.path.join(savepath,"state_at_all_times.csv"),'a') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
#         dict_writer.writeheader()
        dict_writer.writerow(state_snap)
        print("[INFO] appended state for time step: "+ str(state_snap["t"]))
        output_file.close()

# print("[INFO] CSV created.")


# In[8]:


# save_field(field_vals_at_time(os.path.join("wingMotion2D_pimpleFoam/","0.26")))
# save_state(state_vals_at_time(os.path.join("wingMotion2D_pimpleFoam/","0.26")))


# In[ ]:

# print(state_snaps[-5:])


# In[ ]:
