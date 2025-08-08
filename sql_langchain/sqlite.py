# import sqlite 
import sqlite3

# create a connection 
connection = sqlite3.connect("student.db")

#create a cursor object
cursor = connection.cursor()

table_info="""
create table newstudents(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

## Insert some more records
cursor.execute('''Insert Into newstudents values('Krish','Data Science','A',90)''')
cursor.execute('''Insert Into newstudents values('John','Data Science','B',100)''')
cursor.execute('''Insert Into newstudents values('Mukesh','Data Science','A',86)''')
cursor.execute('''Insert Into newstudents values('Jacob','DEVOPS','A',50)''')
cursor.execute('''Insert Into newstudents values('Dipesh','DEVOPS','A',35)''')

# now let's display all the records that have been inserted
data = cursor.execute("SELECT * FROM newstudents")
for row in data:
    print(row)

#now close the connection
connection.commit()
connection.close()