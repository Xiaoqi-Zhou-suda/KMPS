import numpy as np
import os.path
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
import geone as gn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from PIL import Image
TI=np.array(Image.open('./Picture1.png'))[:,:,0:3]
def Bilinear_interpolation(image, dstH, dstW):
    scrH, scrW, _=image.shape
    image=np.pad(image, ((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH, dstW, 3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx = (i+1) * (scrH / dstH) - 1
            scry = (j+1) * (scrW / dstW) - 1
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            retimg[i, j] = (1 - u) * (1 - v) * image[x, y] + u * (1 - v) * image[x + 1, y] + (1 - u) * v * image[
                x, y + 1] + u * v * image[x + 1, y + 1]
    return retimg

size=400
mage_inter=Bilinear_interpolation(TI,size,100)
type=np.zeros((size,100))
mage_inter=mage_inter[:,:,2]
type[mage_inter==[142]]=1
# type[mage_inter==[141]]=1
type[mage_inter==[204]]=2
type[mage_inter==[214]]=3
type[mage_inter==[247]]=4
type[mage_inter==[230]]=5
type[mage_inter==[153]]=6
type[mage_inter==[173]]=7
a=np.argwhere(type==0)
for i in range(a.shape[0]):
    type[a[i][0],a[i][1]]=type[a[i][0]-1,a[i][1]]

training_image=type
cohesion=np.zeros(type.shape)
friction=np.zeros(type.shape)
cohesion[training_image==1]=22
friction[training_image==1]=13
cohesion[training_image==2]=55
friction[training_image==2]=10
cohesion[training_image==3]=30
friction[training_image==3]=15
cohesion[training_image==4]=10
friction[training_image==4]=22
cohesion[training_image==5]=3
friction[training_image==5]=30
cohesion[training_image==6]=33
friction[training_image==6]=15
cohesion[training_image==7]=50
friction[training_image==7]=12

search_radius_x=[10,8,5,4]
search_radius_y=[40,32,20,16]
search_x=search_radius_x[0]
search_y=search_radius_y[0]
nx,ny,nz=int(100-2*search_x), int(size-2*search_y),1
dx,dy,dz=1, 40/size, 1.0
ox,oy,oz=0.0,0.0,0.0
sim_grid=np.zeros((nx*ny,7))
simulated_ratio=np.sum(np.array(sim_grid[:,2]!=0).astype(int))/(nx*ny)
t1=time.time()
fract_of_ti_to_scan=0.8
sim_path=np.random.permutation(nx*ny)
ti_path=np.random.permutation(training_image.shape[0])
sizesim=sim_path.shape[0]
progress=0
tinod=0

#reducing the size of the ti to avoid scanning outside of it
ti_size1=training_image.shape[0]-2*search_y
ti_size2=int(training_image.shape[1])-2*search_x

#creating empty simulation with a wrapping of NaNs of the size

simul=np.full(((ny+2*search_y),(nx+2*search_x)),np.nan)
cohesion_sim=np.full(((ny+2*search_y),(nx+2*search_x)),np.nan)
friction_sim=np.full(((ny+2*search_y),(nx+2*search_x)),np.nan)

path_ti=np.random.permutation(ti_size1*ti_size2)
path_sim=np.random.permutation(nx*ny)
# path_sim=np.random.permutation(simul.shape[0]*simul.shape[1])
sizesim=path_sim.shape[0]
progress=0
tinod=0
simul[int(1 / 40 * size), 10:90] = 1
simul[int(1 / 40 * size),[10,60]] = 1
cohesion_sim[int(1 / 40 * size), [10,60]]=22
friction_sim[int(1 / 40 * size), [10,60]]=13

simul[int(4 / 40 * size), 10:80] = 2
simul[int(4 / 40 * size), [10,60]] = 2
cohesion_sim[int(4 / 40 * size), [10,60]]=55
friction_sim[int(4 / 40 * size), [10,60]]=10

simul[int(10 / 40 * size), 10:80] = 3
simul[int(10 / 40 * size), [10,60]] = 3
cohesion_sim[int(10 / 40 * size), [10,60]]=30
friction_sim[int(10 / 40 * size), [10,60]]=15

simul[int(20 / 40 * size), 10:90] = 4
simul[int(20 / 40 * size), [10,60]] = 4
cohesion_sim[int(20 / 40 * size), [10,60]]=10
friction_sim[int(20 / 40 * size), [10,60]]=22

simul[int(30 / 40 * size), 10:70] = 5
simul[int(30 / 40 * size), [10,60]] = 5
cohesion_sim[int(30 / 40 * size), [10,60]]=3
friction_sim[int(30 / 40 * size), [10,60]]=30

simul[int(35 / 40 * size), 10:80] = 6
simul[int(35 / 40 * size), [10,60]] = 6
cohesion_sim[int(35 / 40 * size), [10,60]]=33
friction_sim[int(35 / 40 * size), [10,60]]=15

simul[int(37 / 40 * size), 10:80] = 7
simul[int(37 / 40 * size), [10,60]] = 7
cohesion_sim[int(37 / 40 * size), [10,60]]=50
friction_sim[int(37 / 40 * size), [10,60]]=12

for simnod in range(0, sizesim):
    progress_current = np.ceil(simnod * 100 / sizesim)
    if progress_current > progress:
        progress = progress_current
        print('the {} of path node completed '.format(progress))
    # fine the node in the simulation grid
    xsim = np.floor(path_sim[simnod] / ny)
    ysim = path_sim[simnod] - xsim * ny

    ratio = progress // 25
    if progress == 100:
        ratio = 3
    search_x = search_radius_x[0 + int(ratio)]
    search_y = search_radius_y[0 + int(ratio)]
    sim_y = search_y
    sim_x = search_x
    point_sim = [int(ysim + search_radius_y[0]), int(xsim + search_radius_x[0])]
    # point_sim=[int(ysim+search_y),int(xsim+search_x)]
    # define data event at simulated point
    data_event_sim = simul[int(point_sim[0] - sim_y):int(point_sim[0] + sim_y),
                     int(point_sim[1] - sim_x):int(point_sim[1] + sim_x)]
    data_event_cohesion_sim = cohesion_sim[int(point_sim[0] - sim_y):int(point_sim[0] + sim_y),
                              int(point_sim[1] - sim_x):int(point_sim[1] + sim_x)]
    data_event_friction_sim = friction_sim[int(point_sim[0] - sim_y):int(point_sim[0] + sim_y),
                              int(point_sim[1] - sim_x):int(point_sim[1] + sim_x)]

    # scan the ti
    mindist = np.inf
    tries = 0
    max_scan = path_ti.shape[0] * fract_of_ti_to_scan

    # reduce the data event to its informed nodes
    no_data_indicator = np.isfinite(data_event_sim)
    no_cohesion_indicator = np.isfinite(data_event_cohesion_sim)
    no_friction_indicator = np.isfinite(data_event_friction_sim)
    data_event_sim = data_event_sim[no_data_indicator]
    data_event_cohesion_sim = data_event_cohesion_sim[no_cohesion_indicator]
    data_event_friction_sim = data_event_friction_sim[no_friction_indicator]

    while 1 == 1:
        tinod = tinod + 1
        tries = tries + 1
        # if arriving at the end of the path,restart
        if tinod >= path_ti.shape[0]:
            tinod = 1

        # fine the point in the ti
        xti = np.floor(path_ti[tinod] / ti_size1)
        yti = path_ti[tinod] - (xti) * ti_size1

        # find scanned point and data event
        point_ti = [yti + search_radius_y[0], xti + search_radius_x[0]]
        # point_ti=[yti+search_y,xti+search_x]
        data_event_ti = training_image[int(point_ti[0] - sim_y):int(point_ti[0] + sim_y),
                        int(point_ti[1] - sim_x):int(point_ti[1] + sim_x)]
        data_event_cohesion_ti = cohesion[int(point_ti[0] - sim_y):int(point_ti[0] + sim_y),
                                 int(point_ti[1] - sim_x):int(point_ti[1] + sim_x)]
        data_event_friction_ti = friction[int(point_ti[0] - sim_y):int(point_ti[0] + sim_y),
                                 int(point_ti[1] - sim_x):int(point_ti[1] + sim_x)]

        # if template is totally unknown
        if np.sum(no_data_indicator[:]) == 0:
            simul[int(point_sim[0]), int(point_sim[1])] = training_image[int(point_ti[0]), int(point_ti[1])]
            cohesion_sim[int(point_sim[0]), int(point_sim[1])] = cohesion[int(point_ti[0]), int(point_ti[1])]
            friction_sim[int(point_sim[0]), int(point_sim[1])] = friction[int(point_ti[0]), int(point_ti[1])]
            break

        # fine the data event at this pount in ti
        data_event_ti = data_event_ti[no_data_indicator]
        data_event_cohesion_ti = data_event_cohesion_ti[no_cohesion_indicator]
        data_event_friction_ti = data_event_friction_ti[no_friction_indicator]

        # evaluate the distance
        distance = np.sum((data_event_sim != data_event_ti).astype(int))
        distance_cohesion = np.sum((data_event_cohesion_sim != data_event_cohesion_ti).astype(int))
        distance_friction = np.sum((data_event_friction_sim != data_event_friction_ti).astype(int))
        # distance = distance + distance_cohesion + distance_friction
        # distance = distance / 3
        thresh = 4 / (ratio + 1)
        if distance <= thresh:
            simul[int(point_sim[0]), int(point_sim[1])] = training_image[int(point_ti[0]), int(point_ti[1])]
            cohesion_sim[int(point_sim[0]), int(point_sim[1])] = cohesion[int(point_ti[0]), int(point_ti[1])]
            friction_sim[int(point_sim[0]), int(point_sim[1])] = friction[int(point_ti[0]), int(point_ti[1])]

            break

        if distance < mindist:
            mindist = distance
            bestpoint = point_ti

        if tries > max_scan:
            simul[int(point_sim[0]), int(point_sim[1])] = training_image[int(bestpoint[0]), int(bestpoint[1])]
            cohesion_sim[int(point_sim[0]), int(point_sim[1])] = cohesion[int(point_ti[0]), int(point_ti[1])]
            friction_sim[int(point_sim[0]), int(point_sim[1])] = friction[int(point_ti[0]), int(point_ti[1])]
            break

t2 = time.time()
print('{} time in total'.format(t2 - t1))
with open('./timecost.txt','a') as f:
    f.write(str(t2-t1)+'\n')

simul_pro=simul[search_radius_y[0]:search_radius_y[0]+ny,search_radius_x[0]:search_radius_x[0]+nx]
simul_pro=simul_pro.reshape(-1)
if os.path.exists('./simulation_vary(TI={}).txt'.format(size)):
    os.remove('./simulation_vary(TI={}).txt'.format(size))
with open('./simulation_vary(TI={}).txt'.format(size),'w') as f:
    f.write('{} {} {} {} {} {} {} {} {}\n'.format(nx,ny,1,1,dy,1,0.0,0.0,0.0))
    f.write('1\n')
    f.write('type\n')
    for i in range(0,simul_pro.shape[0]):
        f.write('{}\n'.format(int(simul_pro[i])))

mse=mean_squared_error(training_image[search_radius_y[0]:search_radius_y[0]+ny,search_radius_x[0]:search_radius_x[0]+nx].reshape(-1),simul_pro)
r2=r2_score(training_image[search_radius_y[0]:search_radius_y[0]+ny,search_radius_x[0]:search_radius_x[0]+nx].reshape(-1),simul_pro)
rmse=np.sqrt(mse)
accuracy=np.mean(training_image[search_radius_y[0]:search_radius_y[0]+ny,search_radius_x[0]:search_radius_x[0]+nx].reshape(-1)==simul_pro)
print(mse,r2,rmse,accuracy)
with open('./metric.txt','a') as f:
    f.write('\n')
    f.write('MSE: {}, R2: {}, RMSE:{},accuracy:{}'.format(mse,r2,rmse,accuracy)+'\n')