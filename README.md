CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yu Sun 
* [LinkedIn](https://www.linkedin.com/in/yusun3/)
* Tested on: Tested on: Windows 10 , i7-6700HQ CPU @ 2.60GHz × 8 , GeForce GTX 960M/PCIe/SSE2, 7.7GB Memory (Personal Laptop)

## Introduction

In this project, stream compaction is implemented using the traditional CPU approach, a naive CUDA based scan algorithm and a more efficient algorithm that performs the scan in place. This project can be used as a base work for many useful things like path tracer. 

Stream compaction usually includes two processes: scan followed by compaction. 

Below is a visualization of how scan and compaction works:
![](images/)

For CPU implementation, scan is basically just an for loop iterating through all the elements and produce the outputs.
![](images/)

For the naive GPU implementation, scan is done using two ping-pong buffers since the algorithm requires inplace updates of the array.
![](images/)

For the efficient GPU implementation, scan is done smartly using up-sweep and down-sweep, which reduces the amount of computation significantly. 
![](images/)

## Performance Analysis 
The performance of these three different algorithms are shown in the diagram below.
i[](images/)



 Output from test Program 
 ```
 
 ```

