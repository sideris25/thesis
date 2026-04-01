import pygimli as pg 
import matplotlib.pyplot as plt
import numpy as np
from pygimli.physics import ert
import pygimli.meshtools as mt
from scipy.interpolate import griddata
import pandas as pd
import concurrent.futures
import time
import os

#geometry
world=mt.createWorld(start=[-1000,0],end=[1500,-1000],worldMarker=1)

poly=mt.createRectangle(start=[0, -140], end=[500, 0], marker=1)

interfaces = []

for i in range(9):

    coords = np.loadtxt(f'shape_{i}.txt')
    line = mt.createPolygon(coords, isClosed=False)
    interfaces.append(line)

geom = mt.mergePLC([world] + interfaces + [poly])

geom.addRegionMarker(pos=[0, -1], marker=2)
geom.addRegionMarker(pos=[0, -6], marker=3)
geom.addRegionMarker(pos=[0, -12], marker=4)
geom.addRegionMarker(pos=[0, -23], marker=5)
geom.addRegionMarker(pos=[0, -40], marker=6)
geom.addRegionMarker(pos=[0, -60], marker=7)
geom.addRegionMarker(pos=[0, -80], marker=8)
geom.addRegionMarker(pos=[0, -100], marker=9)
geom.addRegionMarker(pos=[0, -140], marker=10)

mesh_cut = mt.createMesh(geom, quality=34)

x_min,x_max= 0,500
y_min,y_max= -140,0


#scheme and mesh
spacing=25
scheme=ert.createData(
    elecs=np.linspace(start=0,stop=500,num=len(np.arange(0,501,spacing)))
    , schemeName='dd')

for p in scheme.sensors():
    geom.createNode(p)
    geom.createNode(p-[0,0.1]) # create 2  nodes for FTDT
mesh_fwr=mt.createMesh(geom,quality=34)

#sample production (rhoa)
def generate_sample(sample_id):

    np.random.seed() # !!!! important for multicore processing, otherwise all samples will be the same
    
    res_values=np.random.randint(high=500,low=5,size=10)

    #random res for each layer
    rhomap=[]
    for i in range(10):
        rhomap.append([i + 1, res_values[i]])

    
    try:
        data=ert.simulate(mesh_fwr,scheme=scheme, res=rhomap, noiseLevel=0.01, noiseAbs=1e-6, verbose=False)
        rhoa = np.array(data['rhoa'])

        #Because of nn symmetry requirements, I can't filter the <0 values, so I discard the whole sample.
        if np.any(rhoa <= 0) or np.any(np.isnan(rhoa)):
            return None, None
        
        return rhoa,res_values
    
    except Exception:
        #MISSING RHOA!!!
        return None,None

if __name__ == "__main__":
    num_samples=5000
    output_dir = "nn_dataset"
    os.makedirs(output_dir, exist_ok=True) #Make new dir to save the samples.

    all_X, all_Y = [], []

    start_time = time.time()
    print(f"Generating samples...")

    next_idx=50
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(generate_sample, range(num_samples))
        for i,(X_val,Y_val) in enumerate(results):
            if X_val is not None and Y_val is not None:
                all_X.append(X_val)
                all_Y.append(Y_val)

            idx=i+1
            if idx==next_idx:
                print(f'done {idx}')
                next_idx+=50
       
    dataset_X = np.array(all_X)
    dataset_Y = np.array(all_Y)        
    x_path = os.path.join(output_dir, 'dataset_X.npy')
    y_path = os.path.join(output_dir, 'dataset_Y.npy')

    np.save(x_path, dataset_X)
    np.save(y_path, dataset_Y)

    end_time = time.time()
    print(f"Finished generating samples:{len(all_X)}, Total time: {np.round(end_time - start_time, 2)} seconds.")

