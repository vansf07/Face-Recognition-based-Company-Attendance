![Logo of the project](https://raw.githubusercontent.com/jehna/readme-best-practices/master/sample-logo.png)

# Face Recognition based Company Attendance System
<!-- > Additional information or tagline
 -->
This project is a company attendance system that uses face attendance to recognize employees and marks their entry and exit time in the database. The system effortlessly tracks time for employees with facial recognition, without human intervention or physical validation, as the system can detect and recognize faces autonomously.

## Installing / Getting started

A quick introduction of the minimal setup you need to get a hello world up &
running.

The first thing to do is to clone the repository:
```shell
git clone url
cd Company-Attendance 
```
Create a virtual environment to install dependencies in and activate it:
```shell
virtualenv venv --python=python3.8.10 
source venv/bin/activate 
pip install --upgrade pip==22.1 
```
Then install the dependanceis:
```shell
pip install -r requirements.txt 
```
Once pip has finished downloading the dependencies:
```shell
python manage.py runserver
```
And navigate to http://127.0.0.1:8000/.

