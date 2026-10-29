# The Compression of a Deep Reinforcement Learning Agent onto an Embedded Microcontroller

Machine Learning is a subdivision of Artificial Intelligence (AI) that uses algorithms
to discover patterns in data, enabling a system to” learn” and make predictions or
decisions that improve with experience, without explicit programming.
Reinforcement Learning (RL) is a subdivision of Machine Learning in which a
machine will learn to maximise its cumulative rewards from the feedback it
receives through interactivity with its environment.
Deep Neural Networks have proven to be a highly effective way to implement RL
algorithms, however, the large memory and computational requirements
impeded their deployment onto embedded microcontroller devices. This project
will find the most suitable way to condense the Deep Reinforcement Learning
(DRL) agents onto those devices.
This report will outline the entire development process of the project, from
conception, and research to evaluation of the results. The project was proposed
by Prof HA Engelbrecht, and will be done by Miss AL Beukes as a part of the
Mechatronic Skripsie Project 448

The aim of this project is to compress a trained Deep Reinforcement Learning
algorithm and deploy the agent onto an embedded device, without trading off the
execution speed and performance. The objectives are as follows:
1. Implement Deep Reinforcement Learning algorithm on a desktop
computer.
2. Train the Deep Reinforcement Learning agent on images from a virtual
environment until a baseline performance is established.
3. Compress the policy network and deploy the agent onto an embedded
device.

Deep Reinforcement learning is a powerful but resource-intensive for deploying
to embedded devices. There is a growing demand for autonomous embedded
systems in robotics and IoT devices. It is more feasible and practical for machine
learning algorithms to be on-device systems, rather than cloud-dependent.
TinyML is a subfield of machine learning that focuses on optimising AI models to
be deployed on resource-constrained microcontroller devices. Previous model
compression research primarily focuses on input vector DRL policies and
simulating the model deployment. The embedded microcontroller unit (MCU)
used for this project will be extremely resource-constrained, therefore,
deployment of an image-based DRL policy network onto this device will depict a
meaningful challenge towards the contribution into the TinyML field.

