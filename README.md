# end_to_end_ml_project

## Version Control:

1. First created a ml model that predicts species of iris by taking inputs like SepalLength and more.
2. Next created a index.html file by which a user can give inputs to get output.
3. Created a falsk app which is a framework and integrate the html file and model file

## Docker Containerization: 

1. First step of creating a docker container is to write a dockerfile which contains from ,workdir,copy,run and cmd.
2. To create a docker image command is:
``` bash
docker build -t nikhilreddy001/iris
```
3. To run the docker image command is:
```bash
docker run -p 5000:5000 nikhilreddy001/iris
```
4. Next step is to push the docker container into docker hub
5. First I had loged in to the docker hub by using the command: 
```bash
docker login
```
6. Next step is to push the container command is:
```bash
docker push nikhilreddy001/iris:latest
```
7. You can find the Docker image for this project on [Docker Hub](https://hub.docker.com/r/nikhilreddy001/iris)

## Automated testing:

1. I had used unittest for testing the model in which i had test the model traning function as you see in th test_ml_code.p file
