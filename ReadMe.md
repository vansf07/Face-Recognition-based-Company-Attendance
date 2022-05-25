cd Company-Attendance
pip install --upgrade pip==22.1
sudo pip3 install virtualenv 
virtualenv venv --python=python3.8.10
source venv/bin/activate
pip install -r requirements.txt 
python manage.py runserver
