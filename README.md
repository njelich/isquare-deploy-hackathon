# isquare-deploy-hackathon
Welcome to the Square factory hackathon on model deployment! You will learn how to put your AI models into production using isquare deploy

## Overview

The hackathon will last around two hours (depending on your pace). 

## Pre-requisites
You don't require extensive deep learning knowledge to complete this hackathon. However, there are some pre-requisites. You should have:
- Basic understanding and usage of the unix command line
- Basic understanding of python.
- An python environment on your personal computer, as well as an IDE.
- An internet connection (you probably wouldn't read this if you do not have one)
- A credit card (we will not charge you but it is needed for registration.)

If you are struggling with any of the points of the hackathon, do not hesitate to ask help from one of the experts who should be around if you are participating in the hackathon.

## Introduction and setup 
The following setup steps should be completed before starting:
- Create an [iSquare](app.isquare.ai) account. 
- Clone the project repositories: [The hackathon repo](https://github.com/SquareFactory/isquare-deploy-hackathon) and [the python client for iSquare](https://github.com/SquareFactory/i2-cli)
- Install the python dependencies as well as the python cli

## Deploy
We have provided you with the code of a model which, given an input image, pixelizes the faces on the image and then returns it.

### 1. Play with the model
First, we'd like you to play with the provided scripts and understand how they work:
- Perform an inference using the [provided image](face-pixeliser/imgs/example_01.jpg)
- Take a screenshot with your webcam and see if you can pixelise your face
- [optional] modify the code so that faces are replaced with correctly sized emojis instead of being pixelized.

### 2. Make the model isquare-compatible
Now that you've gotten to know the model a bit, you will have to deploy it and make it available as an "API" using iSquare. Simply follow the steps and explanation given [here](https://docs.isquare.ai/deploy/deploy_with_isquare/1intro) and create the appropriate scripts. You should be able to do this easily on your own, but if you're stuck, we provide the [solution scripts](face-pixeliser/solution). The steps to take can be broken down to:

- Create an iSquare compatible script
- Write a dockerfile for the environment
- Test your build using the i2-cli

### 3. Deploy your model using the web interface
Now that your model is ready to be put into production, use the isquare web interface to deploy your model:
- Deploy the model via github (you can use the scripts provided in the solutions folder or create a branch.)
- Create an access key for your model
- Launch your model
- test your model using the i2-cli (use the `simple.py` script)
- stop your model


### 4. Integrate your model in a production environment
We will provide you with the url and access key to a deployed face pixeliser. 

Use an existing deployed model and interact with the deployed model through cli (streamio webcam) [25 minutes]

install the python cli and package

write a client to stream the webcam (use the provided example file)

SOLUTION PRESENTATION [5 minutes]

Workshop part 2 INTEGRATE WITH FLASK [50 minutes]

Flask backend integration [30 minutes]

Set up the example repository 

Modify the config to the desired model 

Run the frontend and backend services 

Test the model 

SOLUTION PRESENTATION [5 minutes]

Bonus [If your fast]

Modify the config to another model 

Enjoy how the same code makes use of an other model 

Open Question and Feedbak [10 minutes]