# 🐧 `loops` GPU load-balancing library for irregular computations
We propose to build an open-source GPU load-balancing framework for applications that exhibit irregular parallelism. The set of applications and algorithms we consider are fundamental to computing tasks ranging from sparse machine learning, large numerical simulations, and on through to graph analytics. The underlying data and data structures that drive these applications exhibit access patterns that naturally don't map well to the GPU's architecture that is designed with dense and regular access patterns in mind. Prior to the work we present and propose here, the only way to unleash the GPU's full power on these problems has been to workload balance through tightly coupled load-balancing techniques. Our proposed load-balancing abstraction decouples load-balancing from work processing and aims to support both static and dynamic schedules with a programmable interface to implement new load-balancing schedules in the future. With our open-source framework, we hope to not only improve programmers' productivity when developing irregular-parallel algorithms on the GPU but also improve the overall performance characteristics for such applications by allowing a quick path to experimentation with a variety of existing load-balancing techniques. Consequently, we also hope that by separating the concerns of load-balancing from work processing within our abstraction, managing and extending existing code to future architectures becomes easier.

## :musical_note: A little background.
**DARPA** announced [**Software Defined Hardware (SDH)**](https://www.darpa.mil/program/software-defined-hardware), a program that aims "*to build runtime-reconfigurable hardware and software that enables near ASIC performance without sacrificing programmability for data-intensive algorithms.*" **NVIDIA** leading the charge on the program, internally called, [**Symphony**](https://blogs.nvidia.com/blog/2018/07/24/darpa-research-post-moores-law/). My Ph.D. work is a small but important piece of this larger puzzle. The "data-intensive algorithms" part of the program includes domains like Machine Learning, Graph Processing, Sparse-Matrix-Vector algorithms, etc. where there is large amount of data available to be processed. And the problems being addressed are either already based on irregular datastructures and workloads, or are trending towards it (such as sparse machine learning problems). For these irregular workload-computations to be successful, we require efficient load-balancing schemes targetting the specialized hardwares such as the GPUs or Symphony.
- [DARPA Selects Teams to Unleash Power of Specialized, Reconfigurable Computing Hardware](https://www.darpa.mil/news-events/2018-07-24a)

## 🧩 A small (and important) piece of a larger puzzle.
The predominant approach today to addressing irregularity is to build application-dependent solutions. These are not portable between applications. This is a shame because I believe the underlying techniques that are currently used to address irregularity have the potential to be expressed in a generic, portable, powerful way. I intend to build a generic open-source library for load balancing that will expose high-performance, intuitive load-balancing strategies to any irregular-parallel application.

## ⚖️ Load-balancing problem, and a silver lining.
Today's GPUs follow a Single Insruction Multiple Data (SIMD) model, where different work-components (for example a node in a graph) are mapped to a single thread. Each thread runs a copy of program and threads run in parallel (this is a simple explanation, there are other work units in NVIDIA's GPUs such as warps, cooperative thread arrays, streaming multiprocessors etc.). 

> Take graph as an example to discuss the load-balancing problem. Let's say we would like to process neighbors of every single node in a graph, we naïvely map each node to each thread in the GPU. This can result in an extremely inefficient mapping as there could be one node with millions of neighbors while others with only a few neighbors, so the threads that were mapped to nodes with millions of neighbors now end up doing a lot of work while all the other threads idle. The silver lining here is that there are a lot of more intelligent workload mappings that address this problem for different types of graphs or other irregular workloads.

