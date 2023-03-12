# CrowdFlowDetection

![Image](https://github.com/RaviRahar/CrowdFlowDetection/blob/main/results/opticalfb.png)
![Image](https://github.com/RaviRahar/CrowdFlowDetection/blob/main/results/opticalhsv.png)

## Table of contents
- [General Info](#General-Info)
- [To-Do(s)](#TO-DO)
- [Tools Used](#Tools-Used)
- [Contributing](#Contributing)
- [Refrences](#References)

## General Info
We segment every frame of the video into regions of different motions based on the similarity of the neighboring streaklines. Similar streaklines correspond to similar trajectories of particles passing from neighboring pixels over a period of time. Hence, it captures the affinity of current and previous motions at these pixels. First, frame by frame optical flow of the video is computed. Using the optical flow, a set of particles are then moved over the frame to construct the streaklines and the streak flow (not yet implemented). These quantities are used to compute similarity in a 8-connectivity neighborhood. For every pair of pixels i and j, the similarity is computed in terms of streaklines and streak flow (only streaklines for now).

## TO-DO(s)

- [ ] Implement Streakflow
- [ ] Give weightage to streakflow in similarity
- [x] Make separate class
- [x] Implement similarity
- [x] Implement watershed
- [x] Implement Streaklines  
- [x] Use Optical Flow

## Tools Used
- Python
- Pip
- OpenCV
- Numpy

## Contributing


### Setting up environment (for ubuntu)

Install Python

    sudo apt install python3 && sudo apt install python-is-python3
    
Install Pip
    
    sudo apt install python3-pip

Create Virtual Environment
    
    python -m venv env
    source env/bin/activate
    
Install OpenCV and Numpy
    
    pip install opencv-contrib-python numpy
    
### Running for the first time
    
    git clone https://github.com/RaviRahar/CrowdFlowDetection && cd CrowdFlowDetection  
    python main.py  
    
**NOTE:** Do not foget to change video file path in `main.py` to video you want to run it on.

## References
Implementing crowd flow detection using steaklines. Based on:  
[`A Streakline Representation of Flow in Crowded Scenes by Ramin Mehran†, Brian E. Moore‡, Mubarak Shah`][1]

[1]: (https://www.crcv.ucf.edu/papers/eccv2010/Mehran_ECCV_2010.pdf)
