![Logo of the project](https://raw.githubusercontent.com/jehna/readme-best-practices/master/sample-logo.png)

# Face Recognition based Company Attendance System
<!-- > Additional information or tagline
 -->
This project is a company attendance system that uses face attendance to recognize employees and marks their entry and exit time in the database. The system effortlessly tracks time for employees with facial recognition, without human intervention or physical validation, as the system can detect and recognize faces autonomously.

## Installing / Getting started

A quick introduction of the minimal setup you need to get this project up & running.

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

## Admin Login
```bash
username: admin
password: admin123
```

## Features

### Admin
* Register a new Employee. Add photos to dataset via webcam soon after successful registration.
* View attendance of all employees on a particular day (through table and barplot for better comparison)
* View attendance of particular employee during selected duration (through tabel and barplot for better comparison)
* View daily statistics like number of employees present, total number of employees
* View weekly statistics like this week's attendance, last week's attendance (through lineplot)

### Employee
* Mark attendance in-time on company home page
* Mark attendance out-time on company homepage
* View their own attendance (through table and barplot for better comparison)



